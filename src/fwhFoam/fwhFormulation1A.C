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

\*---------------------------------------------------------------------------*/

#include "fwhFormulation1A.H"
#include "error.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * fwhObserverSignal  * * * * * * * * * * * * * * //

Foam::fwhObserverSignal::fwhObserverSignal()
:
    name_("undefined"),
    x_(Zero),
    dt_(GREAT),
    nMin_(0),
    nMax_(-1),
    sumT_(),
    sumL_()
{}


Foam::fwhObserverSignal::fwhObserverSignal
(
    const word& name,
    const point& x,
    const scalar dt
)
:
    name_(name),
    x_(x),
    dt_(dt),
    nMin_(0),
    nMax_(-1),
    sumT_(),
    sumL_()
{}


void Foam::fwhObserverSignal::ensureRange(const label nLo, const label nHi)
{
    if (nMax_ < nMin_)
    {
        // Empty: allocate fresh
        nMin_ = nLo;
        nMax_ = nHi;
        sumT_.setSize(nHi - nLo + 1, Zero);
        sumL_.setSize(nHi - nLo + 1, Zero);
        return;
    }

    if (nLo >= nMin_ && nHi <= nMax_)
    {
        return;
    }

    const label newMin = min(nLo, nMin_);
    const label newMax = max(nHi, nMax_);

    scalarField newT(newMax - newMin + 1, Zero);
    scalarField newL(newMax - newMin + 1, Zero);

    const label off = nMin_ - newMin;
    forAll(sumT_, i)
    {
        newT[off + i] = sumT_[i];
        newL[off + i] = sumL_[i];
    }

    sumT_.transfer(newT);
    sumL_.transfer(newL);
    nMin_ = newMin;
    nMax_ = newMax;
}


void Foam::fwhObserverSignal::addSegment
(
    const scalar t0, const scalar p0T, const scalar p0L,
    const scalar t1, const scalar p1T, const scalar p1L
)
{
    const scalar span = t1 - t0;

    if (span <= VSMALL)
    {
        return;
    }

    // Grid nodes n with t0 < n*dt <= t1
    const label nLo = label(std::floor(t0/dt_)) + 1;
    const label nHi = label(std::floor(t1/dt_ + SMALL));

    if (nHi < nLo)
    {
        return;
    }

    ensureRange(nLo, nHi);

    for (label n = nLo; n <= nHi; ++n)
    {
        const scalar w = (n*dt_ - t0)/span;
        sumT_[n - nMin_] += p0T + w*(p1T - p0T);
        sumL_[n - nMin_] += p0L + w*(p1L - p0L);
    }
}


// * * * * * * * * * * * * * fwhFormulation1A * * * * * * * * * * * * * * * //

inline Foam::scalar Foam::fwhFormulation1A::solveDelay
(
    const vector& d,
    vector& rHat
) const
{
    const scalar U0sq = magSqr(U0_);

    if (U0sq < VSMALL)
    {
        const scalar magd = max(mag(d), VSMALL);
        rHat = d/magd;
        return magd/c0_;
    }

    // Garrick triangle: (c0^2 - U0^2) T^2 + 2 (d.U0) T - |d|^2 = 0
    const scalar a = sqr(c0_) - U0sq;
    const scalar dU = d & U0_;
    const scalar T = (-dU + sqrt(sqr(dU) + a*magSqr(d)))/a;

    const vector rvec = d - U0_*T;
    rHat = rvec/max(mag(rvec), VSMALL);

    return T;
}


Foam::fwhFormulation1A::fwhFormulation1A
(
    const scalar c0,
    const scalar rho0,
    const vector& U0,
    const scalar dtSrc,
    const pointField& Cf,
    const vectorField& nHat,
    const scalarField& dA,
    const List<observerInfo>& observers
)
:
    c0_(c0),
    rho0_(rho0),
    U0_(U0),
    dtSrc_(dtSrc),
    Cf_(Cf),
    nHat_(nHat),
    dA_(dA),
    signals_(),
    Ubuf_(),
    Lbuf_(),
    tau_(),
    nFilled_(0),
    prevPT_(),
    prevPL_(),
    tauPrev_(-GREAT),
    minT_(),
    maxT_(),
    tauFirstEmit_(GREAT),
    tauLastEmit_(-GREAT),
    nSkipped_(0)
{
    if (mag(U0_) >= c0_)
    {
        FatalErrorInFunction
            << "Mean-flow Mach number |U0|/c0 = " << mag(U0_)/c0_
            << " must be subsonic" << exit(FatalError);
    }

    if (dtSrc_ <= VSMALL)
    {
        FatalErrorInFunction
            << "Non-positive source sampling interval " << dtSrc_
            << exit(FatalError);
    }

    signals_.setSize(observers.size());
    forAll(observers, obsi)
    {
        signals_[obsi] = fwhObserverSignal
        (
            observers[obsi].name,
            observers[obsi].position,
            dtSrc_
        );
    }

    const label nObs = signals_.size();
    const label nF = Cf_.size();

    forAll(Ubuf_, li)
    {
        Ubuf_[li].setSize(nF, Zero);
        Lbuf_[li].setSize(nF, Zero);
        tau_[li] = 0;
    }

    prevPT_.setSize(nObs*nF, Zero);
    prevPL_.setSize(nObs*nF, Zero);

    // Propagation delays are constant for static geometry: precompute
    // per-observer extrema for the valid-window bookkeeping
    minT_.setSize(nObs, GREAT);
    maxT_.setSize(nObs, -GREAT);

    forAll(signals_, obsi)
    {
        vector rHat;
        forAll(Cf_, facei)
        {
            const scalar T = solveDelay(signals_[obsi].x() - Cf_[facei], rHat);
            minT_[obsi] = min(minT_[obsi], T);
            maxT_[obsi] = max(maxT_[obsi], T);
        }
    }
}


void Foam::fwhFormulation1A::addTimeLevel
(
    const scalar t,
    const scalarField& pPrime,
    const scalarField& rho,
    const vectorField& u
)
{
    const label nF = Cf_.size();

    if (pPrime.size() != nF || rho.size() != nF || u.size() != nF)
    {
        FatalErrorInFunction
            << "Field size mismatch: expected " << nF << " faces, got p:"
            << pPrime.size() << " rho:" << rho.size() << " u:" << u.size()
            << exit(FatalError);
    }

    // Select buffer slot: fill 0,1,2 then rotate
    label slot;
    if (nFilled_ < 3)
    {
        slot = nFilled_;
    }
    else
    {
        // Rotate: discard oldest
        Foam::Swap(Ubuf_[0], Ubuf_[1]);
        Foam::Swap(Ubuf_[1], Ubuf_[2]);
        Foam::Swap(Lbuf_[0], Lbuf_[1]);
        Foam::Swap(Lbuf_[1], Lbuf_[2]);
        tau_[0] = tau_[1];
        tau_[1] = tau_[2];
        slot = 2;
    }

    // Source terms in the medium-fixed frame:
    //   u_B = u - U0,  v_B = -U0  (static surface)
    //   U_i = v_B + (rho/rho0) (u_B - v_B) = -U0 + (rho/rho0) u
    //   L_i = p' nHat + rho u_B (u . nHat)
    vectorField& U = Ubuf_[slot];
    vectorField& L = Lbuf_[slot];

    forAll(U, facei)
    {
        const vector& ui = u[facei];
        const scalar un = ui & nHat_[facei];

        U[facei] = -U0_ + (rho[facei]/rho0_)*ui;
        L[facei] = pPrime[facei]*nHat_[facei] + rho[facei]*(ui - U0_)*un;
    }

    tau_[slot] = t;

    if (nFilled_ < 3)
    {
        ++nFilled_;
    }

    if (nFilled_ == 3)
    {
        // Verify constant sampling interval
        const scalar d1 = tau_[1] - tau_[0];
        const scalar d2 = tau_[2] - tau_[1];

        if (mag(d1 - dtSrc_) > 1e-6*dtSrc_ || mag(d2 - dtSrc_) > 1e-6*dtSrc_)
        {
            FatalErrorInFunction
                << "Non-uniform source sampling: intervals " << d1 << ", "
                << d2 << " differ from configured dt = " << dtSrc_ << nl
                << "fwhFoam requires a constant sampling interval "
                   "(fixed time step, fixed executeInterval)"
                << exit(FatalError);
        }

        emitLevel();
    }
}


void Foam::fwhFormulation1A::emitLevel()
{
    const label nF = Cf_.size();
    const scalar tauMid = tau_[1];
    const scalar inv2dt = 1.0/(2.0*dtSrc_);
    const scalar invC0 = 1.0/c0_;
    const vector M = -U0_*invC0;
    const scalar magSqrM = magSqr(M);
    const scalar fourPi = 4.0*constant::mathematical::pi;

    const bool havePrev = (tauPrev_ > -GREAT/2);

    forAll(signals_, obsi)
    {
        fwhObserverSignal& sig = signals_[obsi];
        const point& xObs = sig.x();
        const label base = obsi*nF;

        for (label facei = 0; facei < nF; ++facei)
        {
            vector rHat;
            const scalar T = solveDelay(xObs - Cf_[facei], rHat);
            const scalar r = c0_*T;

            if (r < VSMALL)
            {
                ++nSkipped_;
                continue;
            }

            const scalar Mr = M & rHat;
            const scalar omr = 1.0 - Mr;
            const scalar invR = 1.0/r;
            const scalar invOmr2 = 1.0/sqr(omr);

            const scalar A1 = invR*invOmr2;
            const scalar A2 = c0_*(Mr - magSqrM)*sqr(invR)*invOmr2/omr;
            const scalar A3 = sqr(invR)*invOmr2;

            const vector& nf = nHat_[facei];
            const vector& Uc = Ubuf_[1][facei];
            const vector& Lc = Lbuf_[1][facei];

            const vector Udot = (Ubuf_[2][facei] - Ubuf_[0][facei])*inv2dt;
            const vector Ldot = (Lbuf_[2][facei] - Lbuf_[0][facei])*inv2dt;

            const scalar Un = Uc & nf;
            const scalar Udotn = Udot & nf;
            const scalar Lr = Lc & rHat;
            const scalar Ldotr = Ldot & rHat;
            const scalar LM = Lc & M;

            const scalar w = dA_[facei]/fourPi;

            const scalar pT = w*rho0_*(Udotn*A1 + Un*A2);
            const scalar pL =
                w*(Ldotr*A1*invC0 + (Lr - LM)*A3 + Lr*A2*invC0);

            const label idx = base + facei;
            const scalar tArr = tauMid + T;

            if (havePrev)
            {
                sig.addSegment
                (
                    tauPrev_ + T, prevPT_[idx], prevPL_[idx],
                    tArr, pT, pL
                );
            }

            prevPT_[idx] = pT;
            prevPL_[idx] = pL;
        }
    }

    if (tauFirstEmit_ > GREAT/2)
    {
        tauFirstEmit_ = tauMid;
    }
    tauLastEmit_ = tauMid;
    tauPrev_ = tauMid;
}


void Foam::fwhFormulation1A::validWindow
(
    const label obsi,
    scalar& tStart,
    scalar& tEnd
) const
{
    if (Cf_.empty())
    {
        // No local faces: neutral values for max/min reductions
        tStart = -GREAT;
        tEnd = GREAT;
        return;
    }

    if (tauFirstEmit_ > GREAT/2)
    {
        tStart = GREAT;
        tEnd = -GREAT;
        return;
    }

    tStart = tauFirstEmit_ + maxT_[obsi];
    tEnd = tauLastEmit_ + minT_[obsi];
}


// ************************************************************************* //
