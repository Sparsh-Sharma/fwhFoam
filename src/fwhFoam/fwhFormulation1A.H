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

    The implementation uses the porous-surface (permeable) form of
    formulation 1A (Di Francescantonio 1997, Brentner & Farassat 1998),

    \f[
        4\pi p'_T(\vec{x},t) = \int_S \left[
            \frac{\rho_0 \dot{U}_n}{r(1-M_r)^2}
          + \frac{\rho_0 U_n \, c_0 (M_r - M^2)}{r^2 (1-M_r)^3}
        \right]_{ret} dS
    \f]
    \f[
        4\pi p'_L(\vec{x},t) = \int_S \left[
            \frac{\dot{L}_r}{c_0 r (1-M_r)^2}
          + \frac{L_r - L_M}{r^2 (1-M_r)^2}
          + \frac{L_r (M_r - M^2)}{r^2 (1-M_r)^3}
        \right]_{ret} dS
    \f]

    with the permeable-surface source terms
    \f[
        U_i = v_i + \frac{\rho}{\rho_0}(u_i - v_i), \qquad
        L_i = p' \hat{n}_i + \rho u_i (u_n - v_n)
    \f]
    where \f$u\f$ is the fluid velocity and \f$v\f$ the surface velocity,
    both relative to the quiescent acoustic medium. For an impermeable
    surface \f$u_n = v_n\f$ and the classical solid-surface form is
    recovered identically; no separate code path is required.

    A uniform mean (wind-tunnel) convection velocity \c U0 is supported by
    a Galilean transformation to the medium-fixed frame: the surface and
    the observers are treated as translating with \f$-U_0\f$, and the
    emission-to-reception time interval follows from the closed-form
    Garrick-triangle solution
    \f[
        T = \frac{-\vec{d}\cdot\vec{U}_0
             + \sqrt{(\vec{d}\cdot\vec{U}_0)^2 + (c_0^2-U_0^2)|\vec{d}|^2}}
            {c_0^2 - U_0^2},
        \qquad \vec{d} = \vec{x}_{obs} - \vec{y}
    \f]
    so no retarded-time root finding is needed. Surface data are supplied
    in the CFD (wind-tunnel) frame; the transformation is internal.

    Current restrictions (v1):
      - static integration surface (no mesh motion),
      - constant source-time sampling interval,
      - subsonic surface Mach number (|U0| < c0).

    Time derivatives of the source terms are evaluated with second-order
    central differences over three stored source-time levels. Each face
    contribution is spread onto a uniform observer time grid (spacing equal
    to the source sampling interval) by exact piecewise-linear resampling
    between consecutive arrival times, which correctly handles the
    Doppler-shifted, non-uniform arrival pattern.

    The class is mesh-independent: it operates on raw face-centre data
    (positions, unit normals, areas and per-step samples of p', rho, u)
    and is shared by the runtime function object Foam::functionObjects::fwh
    and the offline utility fwhSolve.

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

        //- Face centres (static)
        pointField Cf_;

        //- Face unit normals, pointing away from the sources
        //  (towards the observers)
        vectorField nHat_;

        //- Face areas
        scalarField dA_;

        //- Observer signals
        List<fwhObserverSignal> signals_;

        //- Ring buffer: source-time levels of U_i and L_i per face
        FixedList<vectorField, 3> Ubuf_;
        FixedList<vectorField, 3> Lbuf_;

        //- Source times of the buffered levels
        FixedList<scalar, 3> tau_;

        //- Number of levels currently filled (0..3)
        label nFilled_;

        //- Previous emitted contribution per (face,observer):
        //  flattened [obsi*nFaces + facei]
        scalarField prevPT_;
        scalarField prevPL_;

        //- Previous emission source time (tau of last emitted level),
        //  < -GREAT/2 when nothing emitted yet
        scalar tauPrev_;

        //- Min/max propagation delay T over local faces (per observer)
        scalarField minT_;
        scalarField maxT_;

        //- Source time of first emitted level, GREAT when none
        scalar tauFirstEmit_;

        //- Source time of last emitted level
        scalar tauLastEmit_;

        //- Count of skipped (invalid) face-observer evaluations
        label nSkipped_;


    // Private Member Functions

        //- Emission-to-reception delay T and unit radiation vector rHat
        //  for separation d = x_obs - y (CFD frame). Returns T.
        inline scalar solveDelay(const vector& d, vector& rHat) const;

        //- Evaluate and accumulate contributions of the middle buffered
        //  level (index 1 in time-ordered view)
        void emitLevel();


public:

    // Constructors

        //- Construct from ambient state, geometry and observers.
        //  nHat must point away from the enclosed sources.
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

        label nFaces() const noexcept { return Cf_.size(); }
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

        //- Add one source-time level of surface data (CFD frame):
        //  gauge pressure p' = p - p0 [Pa], density rho [kg/m^3] and
        //  fluid velocity u [m/s] at the face centres. For static
        //  impermeable (wall) surfaces pass the wall values (u = wall
        //  velocity = 0 for a fixed wall). Emits the acoustic
        //  contribution of the previous level once three levels are
        //  buffered.
        void addTimeLevel
        (
            const scalar t,
            const scalarField& pPrime,
            const scalarField& rho,
            const vectorField& u
        );

        //- Observer-time window over which the accumulated signal is
        //  complete (all local faces have contributed):
        //  [tauFirstEmit + max(T), tauLastEmit + min(T)] per observer.
        //  Local (unreduced) values.
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
