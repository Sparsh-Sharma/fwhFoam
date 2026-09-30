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
    Foam::fwhFormulation1A

Description
    Farassat formulation 1A of the Ffowcs Williams--Hawkings (FW-H)
    acoustic analogy for permeable and impermeable integration surfaces,
    evaluated with a source-time-dominant ("advanced time") algorithm.

    The engine supports static and MOVING (translating, rotating,
    deforming) integration surfaces through one code path: the surface
    geometry (face centres, unit normals, areas) and the surface velocity
    are carried per source-time level, and the source-motion terms of
    formulation 1A --- \f$U\,\dot{\hat n}\f$, \f$\dot M\f$ and the
    \f$r\dot M_r\f$ contributions --- are evaluated by central differences
    over three buffered levels. For a static surface the per-level
    geometry is identical and those terms vanish identically, recovering
    the static formulation with no separate code.

    Thickness and loading pressures (Brentner & Farassat 1998):
    \f[
        4\pi p'_T = \int_S \Big[
            \frac{\rho_0(\dot U_n + U_{\dot n})}{r(1-M_r)^2}
          + \frac{\rho_0 U_n (r\dot M_r + c_0 M_r - c_0 M^2)}{r^2(1-M_r)^3}
        \Big]_{ret} dS
    \f]
    \f[
        4\pi p'_L = \int_S \Big[
            \frac{\dot L_r}{c_0 r(1-M_r)^2}
          + \frac{L_r - L_M}{r^2(1-M_r)^2}
          + \frac{L_r (r\dot M_r + c_0 M_r - c_0 M^2)}{c_0 r^2(1-M_r)^3}
        \Big]_{ret} dS
    \f]
    with the permeable-surface source vectors (velocities relative to the
    quiescent medium; \f$u\f$ fluid, \f$v\f$ surface)
    \f[
        U_i = v_i + \frac{\rho}{\rho_0}(u_i - v_i), \qquad
        L_i = p'\hat n_i + \rho u_i (u_n - v_n).
    \f]
    A uniform mean (wind-tunnel) flow \c U0 enters by a Galilean shift to
    the medium frame; the emission delay follows the closed-form
    Garrick-triangle relation, evaluated per level with the instantaneous
    source position, so uniformly convected AND arbitrarily moving
    surfaces need no retarded-time root finding.

    Restrictions: subsonic surface motion relative to the medium
    (\f$M_r < 1\f$ enforced with a guard) and a constant source sampling
    interval.

SourceFiles
    fwhFormulation1A.C

\*---------------------------------------------------------------------------*/

#ifndef fwhFormulation1A_H
#define fwhFormulation1A_H

#include "scalarField.H"
#include "vectorField.H"
#include "pointField.H"
#include "FixedList.H"
#include "List.H"
#include "word.H"
#include "point.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class fwhObserverSignal Declaration
\*---------------------------------------------------------------------------*/

//- Accumulated acoustic pressure time series at one observer position,
//  stored on a uniform absolute time grid t_n = n*dt (n integer).
class fwhObserverSignal
{
    // Private Data

        word name_;
        point x_;

        //- Grid spacing (source sampling interval)
        scalar dt_;

        //- Inclusive integer index range of allocated grid nodes;
        //  empty when nMax_ < nMin_
        label nMin_;
        label nMax_;

        //- Accumulated thickness and loading pressure at grid nodes
        scalarField sumT_;
        scalarField sumL_;


    // Private Member Functions

        //- Ensure grid nodes [nLo,nHi] are allocated (grow with zero fill)
        void ensureRange(const label nLo, const label nHi);


public:

    // Constructors

        //- Default construct (invalid; for container resizing only)
        fwhObserverSignal();

        fwhObserverSignal(const word& name, const point& x, const scalar dt);


    // Member Functions

        const word& name() const noexcept { return name_; }
        const point& x() const noexcept { return x_; }
        scalar dt() const noexcept { return dt_; }
        label nMin() const noexcept { return nMin_; }
        label nMax() const noexcept { return nMax_; }
        const scalarField& sumT() const noexcept { return sumT_; }
        const scalarField& sumL() const noexcept { return sumL_; }

        //- Add the linear segment (t0,p0) -> (t1,p1) of one face's
        //  contribution onto all grid nodes n with t0 < n*dt <= t1
        void addSegment
        (
            const scalar t0, const scalar p0T, const scalar p0L,
            const scalar t1, const scalar p1T, const scalar p1L
        );
};


/*---------------------------------------------------------------------------*\
                      Class fwhFormulation1A Declaration
\*---------------------------------------------------------------------------*/

class fwhFormulation1A
{
public:

    //- Observer definition (name and position in the CFD frame)
    struct observerInfo
    {
        word name;
        point position;
    };


private:

    // Private Data

        //- Ambient speed of sound [m/s]
        scalar c0_;

        //- Ambient density [kg/m^3]
        scalar rho0_;

        //- Uniform mean-flow (wind-tunnel) velocity in the CFD frame [m/s]
        vector U0_;

        //- Source-time sampling interval [s]
        scalar dtSrc_;

        //- Number of surface faces
        label nFaces_;

        //- Observer signals
        List<fwhObserverSignal> signals_;

        //- Ring buffers over three source-time levels
        FixedList<vectorField, 3> Ubuf_;   // mass-flux velocity U_i
        FixedList<vectorField, 3> Lbuf_;   // loading vector L_i
        FixedList<pointField, 3> Cfbuf_;   // face centres (CFD frame)
        FixedList<vectorField, 3> nbuf_;   // unit normals
        FixedList<scalarField, 3> dAbuf_;  // face areas
        FixedList<vectorField, 3> vBbuf_;  // surface velocity, medium frame

        //- Source times of the buffered levels
        FixedList<scalar, 3> tau_;

        //- Number of levels currently filled (0..3)
        label nFilled_;

        //- Previous emitted contribution and arrival time per
        //  (face,observer): flattened [obsi*nFaces + facei]
        scalarField prevPT_;
        scalarField prevPL_;
        scalarField prevArr_;

        //- Whether anything has been emitted yet
        bool emitted_;

        //- Per-observer completeness bookkeeping:
        //  max over faces of the FIRST arrival, min over faces of the
        //  LAST (current) arrival
        scalarField firstArrMax_;
        scalarField lastArrMin_;

        //- Static reference geometry (used when addTimeLevel is called
        //  without per-level geometry)
        pointField Cf0_;
        vectorField nHat0_;
        scalarField dA0_;

        //- Count of skipped (invalid) face-observer evaluations
        label nSkipped_;


    // Private Member Functions

        //- Emission-to-reception delay T and unit radiation vector rHat
        //  for separation d = x_obs - y (CFD frame). Returns T.
        inline scalar solveDelay(const vector& d, vector& rHat) const;

        //- Fill level slot with source terms and geometry
        void fillLevel
        (
            const label slot,
            const scalar t,
            const scalarField& pPrime,
            const scalarField& rho,
            const vectorField& u,
            const pointField& Cf,
            const vectorField& nHat,
            const scalarField& dA,
            const vectorField& vSurf
        );

        //- Evaluate and accumulate contributions of the middle level
        void emitLevel();


public:

    // Constructors

        //- Construct from ambient state, reference geometry and
        //  observers. nHat points away from the enclosed sources.
        fwhFormulation1A
        (
            const scalar c0,
            const scalar rho0,
            const vector& U0,
            const scalar dtSrc,
            const pointField& Cf,
            const vectorField& nHat,
            const scalarField& dA,
            const List<observerInfo>& observers
        );


    // Member Functions

        label nFaces() const noexcept { return nFaces_; }
        label nObservers() const noexcept { return signals_.size(); }
        const List<fwhObserverSignal>& signals() const noexcept
        {
            return signals_;
        }

        scalar c0() const noexcept { return c0_; }
        scalar rho0() const noexcept { return rho0_; }
        const vector& U0() const noexcept { return U0_; }
        scalar dtSrc() const noexcept { return dtSrc_; }
        label nSkipped() const noexcept { return nSkipped_; }

        //- Add one source-time level on the STATIC reference geometry
        //  (surface velocity zero). p' = p - p0 [Pa], rho [kg/m^3],
        //  fluid velocity u [m/s], all in the CFD frame.
        void addTimeLevel
        (
            const scalar t,
            const scalarField& pPrime,
            const scalarField& rho,
            const vectorField& u
        );

        //- Add one source-time level of a MOVING surface: per-level face
        //  centres, unit normals, areas and surface velocity vSurf (CFD
        //  frame). Face ordering must be consistent across levels.
        void addTimeLevel
        (
            const scalar t,
            const scalarField& pPrime,
            const scalarField& rho,
            const vectorField& u,
            const pointField& Cf,
            const vectorField& nHat,
            const scalarField& dA,
            const vectorField& vSurf
        );

        //- Observer-time window over which the accumulated signal is
        //  complete (all local faces have contributed). Local
        //  (unreduced) values; neutral for empty ranks.
        void validWindow
        (
            const label obsi,
            scalar& tStart,
            scalar& tEnd
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
