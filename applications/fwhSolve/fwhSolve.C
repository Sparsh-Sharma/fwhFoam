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

Application
    fwhSolve

Description
    Offline Ffowcs Williams--Hawkings solver: computes observer acoustic
    pressure signals from one or more surface-data files in the FWH-DATA
    format, as written by the fwh function object (writeSurfaceData) or
    generated externally (e.g. by the pyfwh Python tools).

    A parallel OpenFOAM run writes one file per processor
    (surfaceData_proc<N>.fwh), each holding that processor's share of the
    integration surface. fwhSolve processes each file independently and
    sums the resulting observer signals; this is exact because the FW-H
    surface integral is linear in the faces. The combined valid time
    window is the intersection of the per-file windows.

    Configuration (default dictionary: fwhSolveDict):

    \verbatim
    // a single file, or a glob with one '*' matching the per-processor set
    dataFile    "postProcessing/fwh1/acousticData/surfaceData_proc*.fwh";
    outputDir   "fwhSolve-output";

    c0          340.29;
    rho0        1.225;
    U0          (0 0 0);
    stride      1;          // optional: use every stride-th record

    observers   { mic090 { position (0 1.0 0); } }
    \endverbatim

    FWH-DATA format (little-endian IEEE doubles):
    \verbatim
    FWH-DATA 1\n
    nFaces <N>\n
    hasRho <0|1>\n
    binary double\n
    END_HEADER\n
    <geometry: N x (cx cy cz nx ny nz area)>
    <records: t, p'[N], u[3N], (rho[N] if hasRho)>
    \endverbatim
    p' is the gauge pressure in Pa, u the fluid velocity in the CFD frame,
    and the normals point away from the sources (towards the observers).

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "IFstream.H"
#include "OFstream.H"
#include "dictionary.H"
#include "OSspecific.H"
#include "fwhFormulation1A.H"
#include <fstream>
#include <string>
#include <map>
#include <utility>

using namespace Foam;

// * * * * * * * * * * * * * * * * Helpers * * * * * * * * * * * * * * * * * //

//- Per-observer accumulator on the (shared) uniform time grid
struct obsAccum
{
    std::map<label, std::pair<double, double>> grid;   // n -> (sumT, sumL)
    scalar tStart = -GREAT;   // combined window start (max over files)
    scalar tEnd = GREAT;      // combined window end   (min over files)
    label nFiles = 0;
};


//- Expand a data-file specification into a list of files. Accepts a
//  literal existing file or a pattern containing a single '*'.
static fileNameList expandFiles(const fileName& spec)
{
    if (Foam::isFile(spec))
    {
        return fileNameList(1, spec);
    }

    const std::string s(spec);
    const auto star = s.find('*');
    if (star == std::string::npos)
    {
        FatalErrorInFunction
            << "Data file " << spec << " not found" << exit(FatalError);
    }

    fileName dir = spec.path();
    if (dir.empty()) dir = ".";
    const std::string pat(spec.name());
    const auto pstar = pat.find('*');
    const std::string pre = pat.substr(0, pstar);
    const std::string suf = pat.substr(pstar + 1);

    fileNameList out;
    for (const fileName& f : readDir(dir, fileName::FILE))
    {
        const std::string nm(f);
        if (nm.size() >= pre.size() + suf.size()
         && nm.compare(0, pre.size(), pre) == 0
         && nm.compare(nm.size() - suf.size(), suf.size(), suf) == 0)
        {
            out.append(dir/f);
        }
    }
    Foam::sort(out);

    if (out.empty())
    {
        FatalErrorInFunction
            << "No files match " << spec << exit(FatalError);
    }
    return out;
}


//- Process one FWH-DATA file and merge its observer signals into accum.
//  Returns the source sampling interval dtSrc.
static scalar processFile
(
    const fileName& path,
    const scalar c0,
    const scalar rho0,
    const vector& U0,
    const label stride,
    const List<fwhFormulation1A::observerInfo>& observers,
    List<obsAccum>& accum
)
{
    std::ifstream is(path.c_str(), std::ios::binary);
    if (!is.good())
    {
        FatalErrorInFunction
            << "Cannot open data file " << path << exit(FatalError);
    }

    label nFaces = -1;
    bool hasRho = true;
    {
        std::string line;
        bool magicOK = false;
        while (std::getline(is, line))
        {
            if (line == "END_HEADER") break;
            if (line.rfind("FWH-DATA", 0) == 0) magicOK = true;
            else if (line.rfind("nFaces ", 0) == 0)
                nFaces = std::stol(line.substr(7));
            else if (line.rfind("hasRho ", 0) == 0)
                hasRho = (std::stoi(line.substr(7)) != 0);
        }
        if (!magicOK || nFaces < 1)
        {
            FatalErrorInFunction
                << path << " is not a valid FWH-DATA file" << exit(FatalError);
        }
    }

    pointField Cf(nFaces);
    vectorField nHat(nFaces);
    scalarField dA(nFaces);
    for (label i = 0; i < nFaces; ++i)
    {
        double rec[7];
        is.read(reinterpret_cast<char*>(rec), sizeof(rec));
        if (!is)
        {
            FatalErrorInFunction
                << "Truncated geometry block in " << path << exit(FatalError);
        }
        Cf[i] = point(rec[0], rec[1], rec[2]);
        nHat[i] = vector(rec[3], rec[4], rec[5]);
        dA[i] = rec[6];
    }

    autoPtr<fwhFormulation1A> fwh;
    scalarField p(nFaces), rho(nFaces, rho0);
    vectorField u(nFaces);
    bool haveFirst = false;
    scalar tFirst = 0;
    scalarField pFirst, rhoFirst;
    vectorField uFirst;
    List<double> buf(max(3*nFaces, nFaces));
    label nRecords = 0;

    while (true)
    {
        double t;
        is.read(reinterpret_cast<char*>(&t), sizeof(double));
        if (!is) break;

        is.read(reinterpret_cast<char*>(buf.data()), nFaces*sizeof(double));
        for (label i = 0; i < nFaces; ++i) p[i] = buf[i];

        is.read(reinterpret_cast<char*>(buf.data()), 3*nFaces*sizeof(double));
        for (label i = 0; i < nFaces; ++i)
            u[i] = vector(buf[3*i], buf[3*i+1], buf[3*i+2]);

        if (hasRho)
        {
            is.read(reinterpret_cast<char*>(buf.data()), nFaces*sizeof(double));
            for (label i = 0; i < nFaces; ++i) rho[i] = buf[i];
        }
        if (!is) break;   // truncated final record

        if ((nRecords++ % stride) != 0) continue;

        if (!fwh)
        {
            if (!haveFirst)
            {
                tFirst = t; pFirst = p; rhoFirst = rho; uFirst = u;
                haveFirst = true;
                continue;
            }
            fwh.reset
            (
                new fwhFormulation1A
                (c0, rho0, U0, t - tFirst, Cf, nHat, dA, observers)
            );
            fwh->addTimeLevel(tFirst, pFirst, rhoFirst, uFirst);
        }
        fwh->addTimeLevel(t, p, rho, u);
    }

    if (!fwh)
    {
        FatalErrorInFunction
            << "Fewer than two usable records in " << path << exit(FatalError);
    }

    const scalar dt = fwh->dtSrc();
    const List<fwhObserverSignal>& sigs = fwh->signals();
    forAll(sigs, obsi)
    {
        const fwhObserverSignal& sig = sigs[obsi];
        obsAccum& a = accum[obsi];

        scalar tS, tE;
        fwh->validWindow(obsi, tS, tE);
        a.tStart = max(a.tStart, tS);   // intersection of windows
        a.tEnd = min(a.tEnd, tE);
        ++a.nFiles;

        const scalarField& sT = sig.sumT();
        const scalarField& sL = sig.sumL();
        for (label n = sig.nMin(); n <= sig.nMax(); ++n)
        {
            const label i = n - sig.nMin();
            auto& cell = a.grid[n];
            cell.first += sT[i];
            cell.second += sL[i];
        }
    }

    Info<< "  " << path.name() << ": " << nFaces << " faces, "
        << nRecords << " records"
        << (fwh->nSkipped() ? "  (" + Foam::name(fwh->nSkipped())
                            + " skipped)" : "") << nl;

    return dt;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Offline FW-H (Farassat 1A) solver for FWH-DATA surface files"
    );
    argList::noParallel();
    argList::noBanner();
    argList::noFunctionObjects();
    argList::noCheckProcessorDirectories();
    argList::addOption("dict", "file", "Configuration dictionary");

    argList args(argc, argv, false, false);

    const fileName dictFile =
        args.getOrDefault<fileName>("dict", "fwhSolveDict");

    IFstream dictStream(dictFile);
    if (!dictStream.good())
    {
        FatalErrorInFunction
            << "Cannot open dictionary " << dictFile << exit(FatalError);
    }
    dictionary dict(dictStream);

    const fileName dataFile(dict.get<fileName>("dataFile"));
    const fileName outputDir
    (
        dict.getOrDefault<fileName>("outputDir", "fwhSolve-output")
    );
    const scalar c0 = dict.get<scalar>("c0");
    const scalar rho0 = dict.get<scalar>("rho0");
    const vector U0 = dict.getOrDefault<vector>("U0", Zero);
    const label stride = dict.getOrDefault<label>("stride", 1);

    List<fwhFormulation1A::observerInfo> observers;
    const dictionary& obsDict = dict.subDict("observers");
    for (const entry& e : obsDict)
    {
        if (e.isDict())
        {
            fwhFormulation1A::observerInfo obs;
            obs.name = e.keyword();
            e.dict().readEntry("position", obs.position);
            observers.append(obs);
        }
    }
    if (observers.empty())
    {
        FatalErrorInFunction << "No observers defined" << exit(FatalError);
    }

    const fileNameList files = expandFiles(dataFile);
    Info<< "Processing " << files.size() << " surface-data file(s)" << nl
        << "c0 = " << c0 << ", rho0 = " << rho0 << ", U0 = " << U0 << nl
        << "Observers: " << observers.size() << nl << endl;

    List<obsAccum> accum(observers.size());
    scalar dt = 0;
    for (const fileName& f : files)
    {
        const scalar dtf = processFile
        (
            f, c0, rho0, U0, stride, observers, accum
        );
        if (dt == 0) dt = dtf;
        else if (mag(dtf - dt) > 1e-6*dt)
        {
            FatalErrorInFunction
                << "Inconsistent sampling interval across files: "
                << dtf << " vs " << dt << exit(FatalError);
        }
    }

    // ------------------------------------------------------------------
    // Write merged observer signals
    // ------------------------------------------------------------------
    mkDir(outputDir);
    forAll(observers, obsi)
    {
        const obsAccum& a = accum[obsi];
        const word& nm = observers[obsi].name;
        const point& x = observers[obsi].position;

        OFstream os(outputDir/("observer_" + nm + ".dat"));
        os.precision(12);
        os  << "# fwhFoam Farassat-1A observer signal (fwhSolve)" << nl
            << "# observer: " << nm << nl
            << "# position: " << x << nl
            << "# c0: " << c0 << "  rho0: " << rho0 << "  U0: " << U0 << nl
            << "# nFiles: " << a.nFiles << nl
            << "# validWindow: " << a.tStart << " " << a.tEnd << nl
            << "# t pPrime pThickness pLoading" << nl;

        for (const auto& kv : a.grid)
        {
            const scalar tn = kv.first*dt;
            const double T = kv.second.first;
            const double L = kv.second.second;
            os  << tn << ' ' << T + L << ' ' << T << ' ' << L << nl;
        }
        Info<< "Wrote " << outputDir/("observer_" + nm + ".dat")
            << " (valid window: " << a.tStart << " -- " << a.tEnd << ")" << nl;
    }

    Info<< nl << "End" << nl << endl;
    return 0;
}


// ************************************************************************* //
