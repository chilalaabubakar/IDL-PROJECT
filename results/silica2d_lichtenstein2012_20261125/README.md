# Two-dimensional silica from Lichtenstein, Heyde and Freund (2012): checks (25 November 2026)

Item 5c of `docs/NPJ_EXTENSIONS.md`. The supplement of L. Lichtenstein, M. Heyde and
H.-J. Freund, "The crystalline-vitreous interface in two dimensional silica", Phys. Rev. Lett.
109, 106101 (2012), lists the top layer of a silica bilayer on Ru(0001) imaged by scanning
tunnelling microscopy: Table S1, the crystalline-vitreous interface of the paper's Fig. 1b;
Table S2, a vitreous region of Fig. S1b. The Si positions were measured; the authors placed
each O at the midpoint of two neighbouring Si atoms and took the O height (0.052 nm) from
density functional theory.

Read with

```bash
pdftotext -layout Supplemental_Material.pdf si.txt
python -I scripts/silica/extract_lichtenstein2012.py si.txt data/silica2d_lichtenstein2012
```

The coordinates themselves are not committed (they come from the journal's supplement; see
`docs/NPJ_EXTENSIONS.md`). `checks.json` is the summary below.

| | Table S1 (interface) | Table S2 (vitreous) |
|---|---|---|
| Si, O atoms | 717, 1,075 | 664, 998 |
| Area | 12.3 x 6.9 nm | 12.2 x 6.6 nm |
| Si-Si bond (an O at its midpoint) | 0.304 +- 0.026 nm | 0.303 +- 0.022 nm (supplement: 0.302 +- 0.022) |
| Si with 3 bonds | 639 (89%) | 585 (88%) |
| Rings of 4, 5, 6, 7, 8, 9 Si | 7, 29, 248, 24, 6, 2 | 11, 80, 127, 46, 21, 2 |

**What it shows.** The tables are read correctly: the bond length matches the supplement,
and the vitreous region has more 5- than 7-rings, as the supplement states (287 rings found,
the compilation of Dirindin and Coslovich, arXiv:2410.10500, counts 290). Every Si with fewer
than three bonds and every O not at a bond midpoint lies within 0.5 nm of the image edge: inside
the image the network is complete, every Si has three Si neighbours.

**What it means for item 5.** These real glasses contain no vacancy- or interstitial-like
point defects in the imaged area; their disorder is in the ring sizes. They can serve as
(i) a test that a finder trained on simulated two-dimensional silica reports no defect in a
real defect-free network (the "whether" of item 4), and (ii) real starting structures into
which defects are put and relaxed with a two-dimensional silica model, as done for the
Kob-Andersen glass. Both need a model of two-dimensional silica, which is not yet chosen.
