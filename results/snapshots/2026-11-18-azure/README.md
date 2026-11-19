# Benchmark snapshot against Azure, 2026-11-18

Pronunciada 1.0.1 on its held-out tests, next to Microsoft Azure Pronunciation Assessment. Of 84 comparisons, Pronunciada is significantly ahead in 40, not significantly different in 37 and significantly behind in 7. The tests where Azure is ahead are listed in SCOREBOARD.md.

| file | what it is |
|---|---|
| [SCOREBOARD.md](SCOREBOARD.md) | every held-out number against Azure, with the tally by test and by language |
| [zipa_baseline.md](zipa_baseline.md) | Pronunciada against the unmodified ZIPA model it is built on |
| [cv_real.md](cv_real.md) | Common Voice recordings judged by volunteers, both systems |
| [languages.md](languages.md) | all ten languages on native FLEURS test speech |
| [ompal.md](ompal.md) | Mandarin learners (OMPAL), both systems |
| [refreeze_2026-11-18.md](refreeze_2026-11-18.md) | what changed between the 2026-11-17 run and this one |
| [MANIFEST.json](MANIFEST.json) | versions, the machine, the Azure settings and the SHA-256 of every file here |
| tables/ | the data behind the reports, with confidence intervals |

How it was measured: every score of Pronunciada was recomputed on 2026-11-18 with the released 1.0.1 code. Azure's figures come from responses cached on 2026-11-15 to 2026-11-17, one request per recording; nothing was sent to Azure for this snapshot. Speed and memory figures in SCOREBOARD.md are from 2026-11-17.

The reports one level up in `results/` are the 2026-11-17 run, made with version 1.0.0. The numbers here differ slightly because 1.0.1 resamples audio differently; refreeze_2026-11-18.md lists the differences. The tally against Azure is the same in both. The check script named in that note is not in the repository at this tag.

Real-learner recordings with expert marks are public for English and Mandarin only. For the other eight languages the comparisons use native recordings with altered prompts and volunteers' votes, as SCOREBOARD.md explains.

To check a file against the manifest: `shasum -a 256 <file>`.
