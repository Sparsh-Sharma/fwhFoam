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
    nFaces_(Cf.size()),
    signals_(),
    Ubuf_(),
    Lbuf_(),
    Cfbuf_(),
    nbuf_(),
    dAbuf_(),
    vBbuf_(),
    tau_(),
    nFilled_(0),
    prevPT_(),
    prevPL_(),
    prevArr_(),
    emitted_(false),
    firstArrMax_(),
    lastArrMin_(),
    Cf0_(Cf),
    nHat0_(nHat),
    dA0_(dA),
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

    forAll(Ubuf_, li)
    {
        Ubuf_[li].setSize(nFaces_, Zero);
        Lbuf_[li].setSize(nFaces_, Zero);
        Cfbuf_[li].setSize(nFaces_, Zero);
        nbuf_[li].setSize(nFaces_, Zero);
        dAbuf_[li].setSize(nFaces_, Zero);
        vBbuf_[li].setSize(nFaces_, Zero);
        tau_[li] = 0;
    }

    prevPT_.setSize(nObs*nFaces_, Zero);
    prevPL_.setSize(nObs*nFaces_, Zero);
    prevArr_.setSize(nObs*nFaces_, Zero);
    firstArrMax_.setSize(nObs, -GREAT);
    lastArrMin_.setSize(nObs, GREAT);
}


void Foam::fwhFormulation1A::fillLevel
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
)
{
    vectorField& U = Ubuf_[slot];
    vectorField& L = Lbuf_[slot];

    forAll(U, facei)
    {
        const vector& ui = u[facei];
        const vector& vi = vSurf[facei];
        const vector& nf = nHat[facei];

        // Medium-frame velocities: uB = u - U0, vB = v - U0
        const vector vB = vi - U0_;
        const vector uB = ui - U0_;
        const scalar unRel = (ui - vi) & nf;     // u_n - v_n (frame invariant)

        U[facei] = vB + (rho[facei]/rho0_)*(ui - vi);
        L[facei] = pPrime[facei]*nf + rho[facei]*uB*unRel;

        vBbuf_[slot][facei] = vB;
    }

    Cfbuf_[slot] = Cf;
    nbuf_[slot] = nHat;
    dAbuf_[slot] = dA;
    tau_[slot] = t;
}


void Foam::fwhFormulation1A::addTimeLevel
(
    const scalar t,
    const scalarField& pPrime,
    const scalarField& rho,
    const vectorField& u
)
{
    const vectorField vZero(nFaces_, Zero);
    addTimeLevel(t, pPrime, rho, u, Cf0_, nHat0_, dA0_, vZero);
}


void Foam::fwhFormulation1A::addTimeLevel
(
    const scalar t,
    const scalarField& pPrime,
    const scalarField& rho,
    const vectorField& u,
    const pointField& Cf,
    const vectorField& nHat,
    const scalarField& dA,
    const vectorField& vSurf
)
{
    if
    (
        pPrime.size() != nFaces_ || rho.size() != nFaces_
     || u.size() != nFaces_ || Cf.size() != nFaces_
     || nHat.size() != nFaces_ || dA.size() != nFaces_
     || vSurf.size() != nFaces_
    )
    {
        FatalErrorInFunction
            << "Field size mismatch: expected " << nFaces_ << " faces"
            << exit(FatalError);
    }

    label slot;
    if (nFilled_ < 3)
    {
        slot = nFilled_;
    }
    else
    {
        Foam::Swap(Ubuf_[0], Ubuf_[1]);   Foam::Swap(Ubuf_[1], Ubuf_[2]);
        Foam::Swap(Lbuf_[0], Lbuf_[1]);   Foam::Swap(Lbuf_[1], Lbuf_[2]);
        Foam::Swap(Cfbuf_[0], Cfbuf_[1]); Foam::Swap(Cfbuf_[1], Cfbuf_[2]);
        Foam::Swap(nbuf_[0], nbuf_[1]);   Foam::Swap(nbuf_[1], nbuf_[2]);
        Foam::Swap(dAbuf_[0], dAbuf_[1]); Foam::Swap(dAbuf_[1], dAbuf_[2]);
        Foam::Swap(vBbuf_[0], vBbuf_[1]); Foam::Swap(vBbuf_[1], vBbuf_[2]);
        tau_[0] = tau_[1];
        tau_[1] = tau_[2];
        slot = 2;
    }

    fillLevel(slot, t, pPrime, rho, u, Cf, nHat, dA, vSurf);

    if (nFilled_ < 3)
    {
        ++nFilled_;
    }

    if (nFilled_ == 3)
    {
        const scalar d1 = tau_[1] - tau_[0];
        const scalar d2 = tau_[2] - tau_[1];

        if (mag(d1 - dtSrc_) > 1e-6*dtSrc_ || mag(d2 - dtSrc_) > 1e-6*dtSrc_)
        {
            FatalErrorInFunction
                << "Non-uniform source sampling: intervals " << d1 << ", "
                << d2 << " differ from configured dt = " << dtSrc_ << nl
                << "fwhFoam requires a constant sampling interval"
                << exit(FatalError);
        }

        emitLevel();
    }
}


void Foam::fwhFormulation1A::emitLevel()
{
    const scalar tauMid = tau_[1];
    const scalar inv2dt = 1.0/(2.0*dtSrc_);
    const scalar invC0 = 1.0/c0_;
    const scalar fourPi = 4.0*constant::mathematical::pi;
    const label nObs = signals_.size();

    const bool first = !emitted_;

    // per-emit completeness tracking
    scalarField arrMin(nObs, GREAT);
    scalarField arrMaxFirst(nObs, -GREAT);

    for (label facei = 0; facei < nFaces_; ++facei)
    {
        const point& y = Cfbuf_[1][facei];
        const vector& nf = nbuf_[1][facei];
        const scalar w = dAbuf_[1][facei]/fourPi;

        const vector ndot = (nbuf_[2][facei] - nbuf_[0][facei])*inv2dt;
        const vector M = vBbuf_[1][facei]*invC0;
        const scalar magSqrM = magSqr(M);
        const vector Mdot =
            (vBbuf_[2][facei] - vBbuf_[0][facei])*inv2dt*invC0;

        const vector& Uc = Ubuf_[1][facei];
        const vector& Lc = Lbuf_[1][facei];
        const vector Udot = (Ubuf_[2][facei] - Ubuf_[0][facei])*inv2dt;
        const vector Ldot = (Lbuf_[2][facei] - Lbuf_[0][facei])*inv2dt;

        const scalar Un = Uc & nf;
        const scalar Undot = (Udot & nf) + (Uc & ndot);

        for (label obsi = 0; obsi < nObs; ++obsi)
        {
            fwhObserverSignal& sig = signals_[obsi];
            const label idx = obsi*nFaces_ + facei;

            vector rHat;
            const scalar T = solveDelay(sig.x() - y, rHat);
            const scalar r = c0_*T;

            const scalar Mr = M & rHat;
            const scalar omr = 1.0 - Mr;

            if (r < VSMALL || omr < 0.02)
            {
                ++nSkipped_;
                continue;
            }

            const scalar invR = 1.0/r;
            const scalar invOmr2 = 1.0/sqr(omr);

            // K = r*Mdot_r + c0*(Mr - M^2)
            const scalar K = r*(Mdot & rHat) + c0_*(Mr - magSqrM);

            const scalar A1 = invR*invOmr2;
            const scalar A2K = K*sqr(invR)*invOmr2/omr;
            const scalar A3 = sqr(invR)*invOmr2;

            const scalar Lr = Lc & rHat;
            const scalar Ldotr = Ldot & rHat;
            const scalar LM = Lc & M;

            const scalar pT = w*rho0_*(Undot*A1 + Un*A2K);
            const scalar pL =
                w*(Ldotr*A1*invC0 + (Lr - LM)*A3 + Lr*A2K*invC0);

            const scalar tArr = tauMid + T;

            if (!first)
            {
                if (tArr > prevArr_[idx])
                {
                    sig.addSegment
                    (
                        prevArr_[idx], prevPT_[idx], prevPL_[idx],
                        tArr, pT, pL
                    );
                }
                else
                {
                    ++nSkipped_;   // non-monotone arrival (near-sonic)
                }
            }

            prevPT_[idx] = pT;
            prevPL_[idx] = pL;
            prevArr_[idx] = tArr;

            arrMin[obsi] = min(arrMin[obsi], tArr);
            if (first)
            {
                arrMaxFirst[obsi] = max(arrMaxFirst[obsi], tArr);
            }
        }
    }

    forAll(signals_, obsi)
    {
        lastArrMin_[obsi] = arrMin[obsi];
        if (first)
        {
            firstArrMax_[obsi] = arrMaxFirst[obsi];
        }
    }

    emitted_ = true;
}


void Foam::fwhFormulation1A::validWindow
(
    const label obsi,
    scalar& tStart,
    scalar& tEnd
) const
{
    if (nFaces_ == 0)
    {
        // No local faces: neutral values for max/min reductions
        tStart = -GREAT;
        tEnd = GREAT;
        return;
    }

    if (!emitted_)
    {
        tStart = GREAT;
        tEnd = -GREAT;
        return;
    }

    tStart = firstArrMax_[obsi];
    tEnd = lastArrMin_[obsi];
}


// ************************************************************************* //
