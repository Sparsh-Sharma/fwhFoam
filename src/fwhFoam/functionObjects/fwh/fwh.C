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

#include "fwh.H"
#include "volFields.H"
#include "interpolation.H"
#include "addToRunTimeSelectionTable.H"
#include "OFstream.H"
#include "Pstream.H"
#include "OSspecific.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(fwh, 0);
    addToRunTimeSelectionTable(functionObject, fwh, dictionary);
}
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

Foam::fileName Foam::functionObjects::fwh::outputDir() const
{
    return
        time_.globalPath()
       /functionObject::outputPrefix
       /name()
       /"acousticData";
}


void Foam::functionObjects::fwh::makeGeometry()
{
    if (sampledMode_)
    {
        surfPtr_->update();

        Cf_ = surfPtr_->Cf();
        const vectorField& Sf = surfPtr_->Sf();
        dA_ = surfPtr_->magSf();

        nHat_.setSize(Sf.size());
        forAll(Sf, i)
        {
            nHat_[i] = Sf[i]/max(dA_[i], VSMALL);
        }

        if (checkOrientation_)
        {
            // Auto-orient a closed surface: the outward normal satisfies
            // sum(Sf . (Cf - centroid)) > 0
            scalar sumA = gSum(dA_);
            vector xRef = gSum(dA_*Cf_)/max(sumA, VSMALL);

            scalar flux = gSum(dA_*(nHat_ & (Cf_ - xRef)));

            if (flux < 0)
            {
                Info<< "fwh: " << name()
                    << ": sampled surface normals point inwards"
                    << " -- flipping" << endl;
                nHat_ = -nHat_;
            }
        }

        if (flipNormals_)
        {
            nHat_ = -nHat_;
        }
    }
    else
    {
        label n = 0;
        for (const label patchi : patchIDs_)
        {
            n += mesh_.boundary()[patchi].size();
        }

        Cf_.setSize(n);
        nHat_.setSize(n);
        dA_.setSize(n);

        label i = 0;
        for (const label patchi : patchIDs_)
        {
            const fvPatch& pp = mesh_.boundary()[patchi];
            const vectorField& pCf = pp.Cf();
            const vectorField& pSf = pp.Sf();
            const scalarField& pMagSf = pp.magSf();

            forAll(pp, fi)
            {
                Cf_[i] = pCf[fi];
                // Boundary Sf points out of the fluid (into the body);
                // the FW-H normal must point into the fluid
                nHat_[i] = -pSf[fi]/max(pMagSf[fi], VSMALL);
                dA_[i] = pMagSf[fi];
                ++i;
            }
        }
    }

    Info<< "fwh: " << name() << ": integration surface with "
        << returnReduce(Cf_.size(), sumOp<label>())
        << " faces, area " << gSum(dA_) << endl;
}


void Foam::functionObjects::fwh::sampleFields
(
    scalarField& pPrime,
    scalarField& rhoF,
    vectorField& uF
)
{
    const auto& p = lookupObject<volScalarField>(pName_);
    const auto& U = lookupObject<volVectorField>(UName_);

    const bool kinematic = (p.dimensions() != dimPressure);

    const volScalarField* rhoPtr = nullptr;
    if (foundObject<volScalarField>(rhoName_))
    {
        rhoPtr = &lookupObject<volScalarField>(rhoName_);
    }

    if (sampledMode_)
    {
        autoPtr<interpolation<scalar>> interpP
        (
            interpolation<scalar>::New(interpolationScheme_, p)
        );
        autoPtr<interpolation<vector>> interpU
        (
            interpolation<vector>::New(interpolationScheme_, U)
        );

        pPrime = surfPtr_->sample(*interpP);
        uF = surfPtr_->sample(*interpU);

        if (rhoPtr)
        {
            autoPtr<interpolation<scalar>> interpRho
            (
                interpolation<scalar>::New(interpolationScheme_, *rhoPtr)
            );
            rhoF = surfPtr_->sample(*interpRho);
        }
        else
        {
            rhoF.setSize(pPrime.size());
            rhoF = rho0_;
        }
    }
    else
    {
        const label n = Cf_.size();
        pPrime.setSize(n);
        rhoF.setSize(n);
        uF.setSize(n);

        label i = 0;
        for (const label patchi : patchIDs_)
        {
            const fvPatchScalarField& pw = p.boundaryField()[patchi];
            const fvPatchVectorField& Uw = U.boundaryField()[patchi];

            const fvPatchScalarField* rhow =
                rhoPtr ? &rhoPtr->boundaryField()[patchi] : nullptr;

            forAll(pw, fi)
            {
                pPrime[i] = pw[fi];
                uF[i] = Uw[fi];
                rhoF[i] = rhow ? (*rhow)[fi] : rho0_;
                ++i;
            }
        }
    }

    // Gauge pressure in Pa
    if (kinematic)
    {
        pPrime = rho0_*(pPrime - pRef_);
    }
    else
    {
        pPrime -= pRef_;
    }
}


void Foam::functionObjects::fwh::openDataFile()
{
    if (dataFilePtr_)
    {
        return;
    }

    // Each processor writes its own raw file holding only its local
    // faces (no MPI): fully parallel-safe and scalable. fwhSolve sums
    // the per-file observer signals, which is exact because the FW-H
    // surface integral is linear in the faces. In serial this is a
    // single surfaceData_proc0.fwh file.
    if (Cf_.empty())
    {
        return;   // nothing to write on this processor
    }

    // mkDir is idempotent and tolerates concurrent creation
    const fileName dir = outputDir();
    mkDir(dir);

    const fileName f = dir/("surfaceData_proc" + Foam::name(Pstream::myProcNo())
                          + ".fwh");

    dataFilePtr_.reset
    (
        new std::ofstream(f.c_str(), std::ios::binary | std::ios::trunc)
    );

    std::ofstream& os = *dataFilePtr_;
    os << "FWH-DATA 1\n"
       << "nFaces " << Cf_.size() << "\n"
       << "hasRho 1\n"
       << "binary double\n"
       << "END_HEADER\n";

    forAll(Cf_, i)
    {
        double rec[7] =
        {
            Cf_[i].x(), Cf_[i].y(), Cf_[i].z(),
            nHat_[i].x(), nHat_[i].y(), nHat_[i].z(),
            dA_[i]
        };
        os.write(reinterpret_cast<const char*>(rec), sizeof(rec));
    }
    os.flush();
}


void Foam::functionObjects::fwh::appendDataRecord
(
    const scalar t,
    const scalarField& pPrime,
    const scalarField& rhoF,
    const vectorField& uF
)
{
    // Per-processor append (local faces only, no MPI)
    if (!dataFilePtr_)
    {
        return;
    }

    std::ofstream& os = *dataFilePtr_;

    const double td = t;
    os.write(reinterpret_cast<const char*>(&td), sizeof(double));

    forAll(pPrime, i)
    {
        const double v = pPrime[i];
        os.write(reinterpret_cast<const char*>(&v), sizeof(double));
    }
    forAll(uF, i)
    {
        double rec[3] = {uF[i].x(), uF[i].y(), uF[i].z()};
        os.write(reinterpret_cast<const char*>(rec), sizeof(rec));
    }
    forAll(rhoF, i)
    {
        const double v = rhoF[i];
        os.write(reinterpret_cast<const char*>(&v), sizeof(double));
    }
    os.flush();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::fwh::fwh
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    c0_(0),
    rho0_(0),
    pRef_(0),
    U0_(Zero),
    pName_("p"),
    UName_("U"),
    rhoName_("rho"),
    sampledMode_(false),
    patchIDs_(),
    surfPtr_(nullptr),
    interpolationScheme_("cellPoint"),
    checkOrientation_(true),
    flipNormals_(false),
    Cf_(),
    nHat_(),
    dA_(),
    observers_(),
    fwhPtr_(nullptr),
    haveFirstSample_(false),
    tFirst_(0),
    pFirst_(),
    rhoFirst_(),
    uFirst_(),
    writeSurfaceData_(false),
    dataFilePtr_(nullptr)
{
    read(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::functionObjects::fwh::read(const dictionary& dict)
{
    fvMeshFunctionObject::read(dict);

    dict.readEntry("c0", c0_);
    dict.readEntry("rho0", rho0_);
    pRef_ = dict.getOrDefault<scalar>("pRef", 0);
    U0_ = dict.getOrDefault<vector>("U0", Zero);

    pName_ = dict.getOrDefault<word>("p", "p");
    UName_ = dict.getOrDefault<word>("U", "U");
    rhoName_ = dict.getOrDefault<word>("rho", "rho");

    writeSurfaceData_ = dict.getOrDefault<bool>("writeSurfaceData", false);

    // Surface
    const dictionary& surfDict = dict.subDict("surface");
    const word surfType(surfDict.get<word>("type"));

    if (surfType == "patches")
    {
        sampledMode_ = false;
        surfPtr_.clear();

        wordRes patchNames(surfDict.get<wordRes>("patches"));
        patchIDs_ = mesh_.boundaryMesh().patchSet(patchNames).sortedToc();

        if (patchIDs_.empty())
        {
            FatalIOErrorInFunction(dict)
                << "No patches matching " << flatOutput(patchNames)
                << exit(FatalIOError);
        }
    }
    else if (surfType == "sampled")
    {
        sampledMode_ = true;
        patchIDs_.clear();

        interpolationScheme_ =
            surfDict.getOrDefault<word>("interpolationScheme", "cellPoint");
        checkOrientation_ =
            surfDict.getOrDefault<bool>("checkOrientation", true);
        flipNormals_ = surfDict.getOrDefault<bool>("flipNormals", false);

        surfPtr_ = sampledSurface::New
        (
            name() + "Surface",
            mesh_,
            surfDict.subDict("surfaceDict")
        );
    }
    else
    {
        FatalIOErrorInFunction(dict)
            << "Unknown surface type '" << surfType
            << "'; valid types: patches, sampled"
            << exit(FatalIOError);
    }

    // Observers
    observers_.clear();
    const dictionary& obsDict = dict.subDict("observers");
    for (const entry& e : obsDict)
    {
        if (e.isDict())
        {
            fwhFormulation1A::observerInfo obs;
            obs.name = e.keyword();
            e.dict().readEntry("position", obs.position);
            observers_.append(obs);
        }
    }

    if (observers_.empty())
    {
        FatalIOErrorInFunction(dict)
            << "No observers defined" << exit(FatalIOError);
    }

    makeGeometry();

    // Reset accumulation state
    fwhPtr_.clear();
    haveFirstSample_ = false;
    dataFilePtr_.reset(nullptr);

    Info<< "fwh: " << name() << ": "
        << (sampledMode_ ? "permeable (sampled)" : "impermeable (patch)")
        << " surface, " << observers_.size() << " observers, c0 = " << c0_
        << ", rho0 = " << rho0_ << ", U0 = " << U0_ << endl;

    return true;
}


bool Foam::functionObjects::fwh::execute()
{
    scalarField pPrime, rhoF;
    vectorField uF;

    sampleFields(pPrime, rhoF, uF);

    const scalar t = time_.timeOutputValue();

    if (writeSurfaceData_)
    {
        openDataFile();
        appendDataRecord(t, pPrime, rhoF, uF);
    }

    if (!fwhPtr_)
    {
        if (!haveFirstSample_)
        {
            tFirst_ = t;
            pFirst_ = pPrime;
            rhoFirst_ = rhoF;
            uFirst_ = uF;
            haveFirstSample_ = true;
            return true;
        }

        const scalar dtSrc = t - tFirst_;

        fwhPtr_.reset
        (
            new fwhFormulation1A
            (
                c0_, rho0_, U0_, dtSrc,
                Cf_, nHat_, dA_, observers_
            )
        );

        fwhPtr_->addTimeLevel(tFirst_, pFirst_, rhoFirst_, uFirst_);

        pFirst_.clear();
        rhoFirst_.clear();
        uFirst_.clear();
    }

    fwhPtr_->addTimeLevel(t, pPrime, rhoF, uF);

    return true;
}


bool Foam::functionObjects::fwh::write()
{
    if (!fwhPtr_)
    {
        return true;
    }

    const scalar dt = fwhPtr_->dtSrc();
    const fileName dir = outputDir();

    if (Pstream::master())
    {
        mkDir(dir);
    }

    const label nSkip = returnReduce(fwhPtr_->nSkipped(), sumOp<label>());
    if (nSkip > 0)
    {
        WarningInFunction
            << "fwh: " << name() << ": " << nSkip
            << " invalid face-observer evaluations were skipped"
            << " (observer too close to the surface?)" << endl;
    }

    const List<fwhObserverSignal>& sigs = fwhPtr_->signals();

    forAll(sigs, obsi)
    {
        const fwhObserverSignal& sig = sigs[obsi];

        // Reduce valid window
        scalar tStart, tEnd;
        fwhPtr_->validWindow(obsi, tStart, tEnd);
        tStart = returnReduce(tStart, maxOp<scalar>());
        tEnd = returnReduce(tEnd, minOp<scalar>());

        // Establish a common global time-grid index range across all
        // processors, then sum each processor's partial signal onto it
        // with fixed-size reductions (robust for any decomposition).
        const bool localEmpty = (sig.nMax() < sig.nMin());

        label gMin =
            returnReduce(localEmpty ? labelMax : sig.nMin(), minOp<label>());
        label gMax =
            returnReduce(localEmpty ? labelMin : sig.nMax(), maxOp<label>());

        if (gMax < gMin)
        {
            continue;   // no processor accumulated anything for this obs
        }

        const label nG = gMax - gMin + 1;
        scalarField gT(nG, Zero);
        scalarField gL(nG, Zero);

        if (!localEmpty)
        {
            const label off = sig.nMin() - gMin;
            const scalarField& sT = sig.sumT();
            const scalarField& sL = sig.sumL();
            forAll(sT, i)
            {
                gT[off + i] = sT[i];
                gL[off + i] = sL[i];
            }
        }

        reduce(gT, sumOp<scalarField>());
        reduce(gL, sumOp<scalarField>());

        if (!Pstream::master())
        {
            continue;
        }

        OFstream os(dir/("observer_" + sig.name() + ".dat"));
        os.precision(12);

        os  << "# fwhFoam Farassat-1A observer signal" << nl
            << "# observer: " << sig.name() << nl
            << "# position: " << sig.x() << nl
            << "# c0: " << fwhPtr_->c0()
            << "  rho0: " << fwhPtr_->rho0()
            << "  U0: " << fwhPtr_->U0() << nl
            << "# validWindow: " << tStart << " " << tEnd << nl
            << "# t pPrime pThickness pLoading" << nl;

        for (label n = gMin; n <= gMax; ++n)
        {
            const scalar tn = n*dt;
            const label i = n - gMin;
            os  << tn << ' ' << gT[i] + gL[i] << ' '
                << gT[i] << ' ' << gL[i] << nl;
        }
    }

    return true;
}


bool Foam::functionObjects::fwh::end()
{
    return write();
}


// ************************************************************************* //
