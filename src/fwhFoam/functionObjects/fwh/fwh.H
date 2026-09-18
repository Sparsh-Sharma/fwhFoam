/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | fwhFoam: Ffowcs Williams-Hawkings acoustics
   \\    /   O peration     | for OpenFOAM
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2026 fwhFoam authors
-------------------------------------------------------------------------------
License
    This file is part of fwhFoam, an extension to OpenFOAM.

    fwhFoam is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    fwhFoam is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with fwhFoam.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::functionObjects::fwh

Group
    grpFieldFunctionObjects

Description
    Runtime Ffowcs Williams--Hawkings (FW-H) acoustic post-processing:
    far-field acoustic pressure at a set of observers, computed from CFD
    surface data on either an impermeable (wall patch) or a permeable
    (sampled) integration surface using Farassat formulation 1A with an
    advanced-time algorithm (see Foam::fwhFormulation1A).

    Works with compressible solvers (p in Pa; rho sampled if available)
    and incompressible solvers (kinematic pressure detected from the
    dimensions of \c p and scaled by \c rho0; rho taken as \c rho0).

Usage
    Example of function object specification:
    \verbatim
    fwh1
    {
        type            fwh;
        libs            (fwhFoam);

        // Ambient conditions
        c0              340.29;         // speed of sound [m/s]
        rho0            1.225;          // ambient density [kg/m^3]
        pRef            0;              // reference (ambient) pressure,
                                        // same units as the p field
        U0              (30 0 0);       // uniform mean-flow velocity
                                        // (wind-tunnel frame), default (0 0 0)

        // Integration surface: either wall patches (impermeable) ...
        surface
        {
            type        patches;
            patches     (cylinder);     // wordRe list
        }

        // ... or a sampled surface (permeable)
        //surface
        //{
        //    type                sampled;
        //    surfaceDict
        //    {
        //        type            sampledTriSurfaceMesh;
        //        surface         fwhSurface.stl;   // constant/triSurface/
        //        source          cells;
        //    }
        //    interpolationScheme cellPoint;  // default
        //    checkOrientation    true;       // closed-surface auto-orient
        //    flipNormals         false;
        //}

        observers
        {
            mic090 { position (0 1.0 0); }
            mic045 { position (0.707 0.707 0); }
        }

        // Optional raw surface-data export for offline use with fwhSolve
        writeSurfaceData false;

        // Optional field names
        p               p;
        U               U;
        rho             rho;

        // Sampling every execute() call; use executeInterval to
        // subsample in time (interval must stay constant)
        executeControl  timeStep;
        executeInterval 1;
        writeControl    writeTime;
    }
    \endverbatim

Note
    - The integration surface must be static; the time step (times
      executeInterval) must be constant.
    - Patch normals are flipped internally so that the FW-H surface
      normal points into the fluid. For sampled closed surfaces the
      orientation is checked and corrected automatically
      (checkOrientation); for open surfaces disable the check and set
      flipNormals so that normals point away from the sources.
    - Observer signals accumulate in memory and are (re)written at each
      write(); output in postProcessing/<name>/acousticData/.
    - Restarts do not resume the accumulated signal; for full control
      use writeSurfaceData with the offline fwhSolve utility.

SourceFiles
    fwh.C

\*---------------------------------------------------------------------------*/

#ifndef functionObjects_fwh_H
#define functionObjects_fwh_H

#include "fvMeshFunctionObject.H"
#include "fwhFormulation1A.H"
#include "sampledSurface.H"
#include "wordRes.H"
#include <fstream>
#include <memory>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                             Class fwh Declaration
\*---------------------------------------------------------------------------*/

class fwh
:
    public fvMeshFunctionObject
{
    // Private Data

        // Ambient state

            scalar c0_;
            scalar rho0_;
            scalar pRef_;
            vector U0_;

        // Field names

            word pName_;
            word UName_;
            word rhoName_;

        // Surface definition

            //- true: sampled (permeable) surface; false: wall patches
            bool sampledMode_;

            //- Patch ids (patch mode)
            labelList patchIDs_;

            //- Sampled surface (sampled mode)
            autoPtr<sampledSurface> surfPtr_;

            word interpolationScheme_;
            bool checkOrientation_;
            bool flipNormals_;

        // Cached (static) integration-surface geometry

            pointField Cf_;
            vectorField nHat_;
            scalarField dA_;

        // Observers

            List<fwhFormulation1A::observerInfo> observers_;

        // Solver core (constructed at the second execute() call, once
        // the sampling interval is known)

            autoPtr<fwhFormulation1A> fwhPtr_;

            //- Cached first sample
            bool haveFirstSample_;
            scalar tFirst_;
            scalarField pFirst_;
            scalarField rhoFirst_;
            vectorField uFirst_;

        // Raw surface-data export

            bool writeSurfaceData_;
            std::unique_ptr<std::ofstream> dataFilePtr_;


    // Private Member Functions

        //- Build integration-surface geometry (Cf_, nHat_, dA_)
        void makeGeometry();

        //- Sample p' (absolute, Pa), rho and u at the surface face centres
        void sampleFields
        (
            scalarField& pPrime,
            scalarField& rhoF,
            vectorField& uF
        );

        //- Open the raw data file and write header + geometry (master)
        void openDataFile();

        //- Append one record to the raw data file
        void appendDataRecord
        (
            const scalar t,
            const scalarField& pPrime,
            const scalarField& rhoF,
            const vectorField& uF
        );

        //- Output directory
        fileName outputDir() const;


public:

    //- Runtime type information
    TypeName("fwh");


    // Constructors

        //- Construct from Time and dictionary
        fwh
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );

        //- No copy construct
        fwh(const fwh&) = delete;

        //- No copy assignment
        void operator=(const fwh&) = delete;


    //- Destructor
    virtual ~fwh() = default;


    // Member Functions

        //- Read the fwh data
        virtual bool read(const dictionary& dict);

        //- Sample surface data and advance the FW-H integration
        virtual bool execute();

        //- Reduce and write the observer signals
        virtual bool write();

        //- Final write on run end
        virtual bool end();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
