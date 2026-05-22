# Rename review — arc_challenge (OLMO)

Source: top-20 features by z-norm area distance from `group_features_by_task_shape_olmo.html`.
Each row reviewed against the 10–15 top-activating samples.

Verdict legend:
- **keep** — description matches samples
- **refine** — description is roughly right but could be tighter/more accurate
- **rename** — description does not match the samples

| fid | dist | verdict | current description | suggested |
|---|---|---|---|---|
| 1190 | 0.934 | keep | Text addressing the user of a personal electronic device. | — |
| 2943 | 1.115 | keep | The word 'former', used to refer to the first of two previously mentioned things. | — |
| 568  | 1.228 | keep | Hyphens used to form compound adjectives. | — (samples skew technical/scientific, but the syntactic claim holds) |
| 619  | 1.311 | refine | The English conjunction "and" \| The German conjunction "und". | **The English conjunction "and".** (no German "und" appears in the top samples — drop the unsupported half) |
| 1758 | 1.349 | keep | Tokens appearing in contexts about winning, receiving, or being present at an awards ceremony or receiving a prize/honor. | — (could shorten to: *Function-word tokens (`the`, `it`) inside awards / prize-giving contexts.*) |
| 748  | 1.359 | keep | Words that denote or are used in contexts of dividing, structuring, or categorizing data or concepts, often in programming or data science. | — |
| 2424 | 1.399 | refine | The TAB character used for paragraph indentation. | **TAB character separating sections/list items in marketing/listicle web text.** (most samples are TABs *between* paragraphs/headings, not paragraph-internal indentation) |
| 1087 | 1.413 | keep | Words within prompts asking the user to share their opinions, thoughts, or preferences. | — |
| 2607 | 1.437 | keep | The comma character used as a separator in source code and other formal languages. | — |
| 798  | 1.438 | keep | The word "into" used in the phrasal verb "to take into account". | — (every sample is take/took/taking + into + account) |
| 1171 | 1.481 | keep | The English function words 'the' and 'it', preceded by a space. | — |
| 2019 | 1.502 | keep | The file extension `.html` appearing at the end of URLs or file paths. | — |
| 2803 | 1.502 | keep | Possessive pronouns used to describe a family member or close friend. | — (samples are almost entirely "his/her/my brother/sister/father/uncle") |
| 2112 | 1.532 | refine | Words from various non-English European languages, including German, Russian, Spanish, and French. | **Words from non-English European languages, predominantly Russian (Cyrillic) and German.** (no Spanish/French in top samples) |
| 1055 | 1.541 | refine | Verbs ending in -ing used as gerunds. | **Gerunds (V-ing), most often following the preposition "by".** (the "by V-ing" construction dominates the top samples) |
| 2646 | 1.543 | keep | Tokens appearing in contexts related to digital marketing, sales, and customer engagement. | — |
| 628  | 1.552 | keep | Numerals used as list item markers. | — (samples are "4." / "8." starting numbered list items) |
| 986  | 1.580 | keep | Plural nouns for entities or phenomena under scientific, technical, or philosophical analysis. | — |
| 2773 | 1.608 | keep | Words used to introduce a person's or entity's role, title, or function. | — (samples: became/be/as + the role) |
| 687  | 1.652 | keep | Verbs meaning to maintain a state or condition. | — (stay/stayed/stays/kept/keep/keeping) |

## Summary

- **17 / 20** descriptions are accurate as-is.
- **3 / 20** could be refined: f619 (drop unsupported German claim), f2424 (TAB role is section separator, not indentation), f2112 (drop Spanish/French claim), f1055 (specialise to "by V-ing").
- No outright renames needed.

## Notes / caveats

- Review is based on the top samples ranked by z-distance to the task curve. A feature description is also expected to cover its broader sample pool; refinements above flag claims that aren't supported in *these* samples, not necessarily across all activations.
- Refinements for f619 and f2112 only narrow the description — the broader meta JSON may still hold supporting samples elsewhere; worth confirming before editing the underlying JSONs.
