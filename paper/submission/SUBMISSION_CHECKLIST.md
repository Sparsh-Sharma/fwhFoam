# SoftwareX submission checklist — fwhFoam

Submit through the SoftwareX Editorial Manager: https://www.editorialmanager.com/softx/

## Files in this folder

| File | Use at submission |
|------|-------------------|
| `fwhFoam_softwarex.tex` | Manuscript source (official SoftwareX OSP template, elsarticle) |
| `fwhFoam_softwarex.pdf` | Compiled manuscript (upload as the main article PDF, or let EM build it from source) |
| `figs/convergence.pdf` | Figure 1 — convergence |
| `figs/cylinder.pdf` | Figure 2 — cylinder Aeolian tone |
| `figs/sigma_cylinder.pdf` | Figure 3 — sigma source map |
| `highlights.txt` | Paste into the "Highlights" step (3–5 items, ≤85 chars each) |
| `cover_letter.md` | Convert to PDF/text; upload as the cover letter |
| `declaration_of_interests.md` | Choose a statement, convert to **.docx**, upload |
| `ai_use_declaration_DRAFT.md` | **Review, edit, and own** — then paste the final statement into the manuscript before the references, and complete the AI-use question in EM |

## Mandatory requirements (from the SoftwareX Guide for Authors)

- [x] Public GitHub repository (mandatory): https://github.com/Sparsh-Sharma/fwhFoam
- [x] Repository has a documented `README.md` and a `LICENSE` file
- [x] Official SoftwareX Original Software Publication template used
- [x] Required Metadata table (C1–C8) completed in the manuscript
- [x] Five mandatory sections present (Motivation and significance;
      Software description; Illustrative examples; Impact; Conclusions)
- [x] Word count within the 4000-word limit (body ≈ 1700 words)
- [x] Figures ≤ 6 (3 used)
- [ ] Declaration of competing interests uploaded as a `.docx`
- [ ] Declaration of generative-AI use finalized and added to the manuscript
- [ ] Funding statement provided (add to Acknowledgements or the EM form)
- [ ] Corresponding-author details confirmed in EM

## Before you click submit

1. **Read the code and the paper end to end and take ownership.** Reviewers
   will install, run, and try to reuse the software; you are accountable for
   every result and claim.
2. Tag a release on GitHub (e.g. `v1.0.0`) so the "current code version" in
   the metadata table points to a fixed, citable state; consider archiving it
   (e.g. via Zenodo) for a DOI.
3. Confirm the repository is public and the README/LICENSE render correctly.
4. Finalize the AI-use declaration (see the draft) and the competing-interest
   and funding statements.

## Rebuild the PDF from source

```bash
cd submission
pdflatex fwhFoam_softwarex.tex && pdflatex fwhFoam_softwarex.tex
```
(Requires a LaTeX distribution with the `elsarticle` class.)
