# Sample recordings: sources and licenses

The nine `.mp3` clips in this folder are the only benchmark recordings in the repository. Each is
an item of a **held-out test split** used in the benchmarks (results/SCOREBOARD.md), re-encoded to
MP3; the matching `.json` is Pronunciada's output for it plus the corpus labels. All three corpora
are licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), which permits
redistribution with attribution; the clips are unmodified apart from the re-encoding.

| file | corpus, split | prompt | corpus item |
|---|---|---|---|
| `en-clean.mp3` | speechocean762, test | It was good for me | `000240010` |
| `en-learner-1.mp3` | speechocean762, test | It was a beautiful day for a float | `014200206` |
| `en-learner-2.mp3` | speechocean762, test | Follow the link below to register | `021160249` |
| `de-native.mp3` | FLEURS `de_de`, test | Parisern sagt man nach, egozentrisch, unhöflich und arrogant zu sein. | test row id `12628233572987997427` |
| `es-native.mp3` | FLEURS `es_419`, test | Los luchadores compañeros de Luna también le rindieron homenaje. | one of the test recordings of this sentence |
| `fr-native.mp3` | FLEURS `fr_fr`, test | Les porcs sont porteurs de la maladie, qui est ensuite transmise à l'homme par les moustiques. | test row id `10991225759022624703` |
| `ja-native.mp3` | FLEURS `ja_jp`, test | そのため、単にラベルとして表記が追加されがちです。 | test row id `15637287613679762593` |
| `ko-native.mp3` | FLEURS `ko_kr`, test | 하지만 여전히 새에는 공룡처럼 보이게 하는 것들이 많습니다. | one of the test recordings of this sentence |
| `zh-learner-1.mp3` | OMPAL, test | 我家有四口人 | a French-L1 learner's test-split reading of this prompt |

Citations:

- **speechocean762**: Junbo Zhang, Zhiwen Zhang, Yongqing Wang, Zhiyong Yan, Qiong Song, Yukai
  Huang, Ke Li, Daniel Povey, Yujun Wang. "speechocean762: An Open-Source Non-native English
  Speech Corpus For Pronunciation Assessment." Interspeech 2021. https://www.openslr.org/101/
- **FLEURS**: Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia,
  Jason Riesa, Clara Rivera, Ankur Bapna. "FLEURS: Few-shot Learning Evaluation of Universal
  Representations of Speech." IEEE SLT 2022. https://huggingface.co/datasets/google/fleurs
- **OMPAL**: Hsieh et al. "OMPAL: Bridging Speech and Learning with an Open-Source Mandarin
  Pronunciation Assessment Corpus for Global Learners." Interspeech 2025,
  doi:10.21437/Interspeech.2025-983. https://github.com/phantomhsieh/OMPAL-corpus

Because these are test-split items, the benchmark numbers include them; they were chosen for
illustration after the benchmarks were run and nothing was tuned on them.
