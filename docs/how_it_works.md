### Pipeline

1. **Parse** each screenplay into timestamped beats: camera shots, actions, sound effects and dialogue lines with their delivery cue.
2. **Detect the genre** of the whole show (drama, thriller, horror, romance or comedy). For each genre, count the share of lines whose text or delivery cue contains one of its marker words. Another genre wins over the default (drama) only if its share is at least 8% and at least 1.2× drama's. The winning genre's `overrides` are merged into the config: for example, a thriller weights danger and the cliffhanger higher and treats "the phone rings" or a countdown as a sting; a comedy adds a *humor* stakes category, counts *Deadpan* or *Mortified* as intensity, and treats an awkward silence or a record scratch as the cut. Drama has no overrides, so the 50-episode input pack (detected as drama) is scored exactly as before. You can also force a genre (`--genre`, or the website's picker).
3. **Score each beat** with four numbers in [0, 1], using only the word lists and weights in `config.yaml`:
   - **I**, intensity, from the delivery cue (*Furious* raises it, *Gentle* lowers it).
   - **S**, stakes. This is a noisy-OR over categories: S = 1 − ∏ₖ (1 − wₖ·[category k present]). The categories are violence, danger, betrayal, reveal, humiliation, romance and defiance.
   - **O**, openness: a question, a line cut off with "…", a threat, or a dramatic sting.
   - **R**, resolution: calm or closing words.
4. **Hook = surprise** and **cliffhanger = suspense**, following Ely, Frankel & Kamenica (2015):
   - H = 0.45·I + 0.55·S (+ a bonus for a short, punchy line, − a penalty for a line that starts mid-sentence)
   - C = 0.40·O + 0.30·S + 0.25·I − 0.45·R
5. **Enumerate every clip** from any beat to any later beat, 20–60 s long. Cuts snap to camera-shot boundaries and never land inside a spoken line.
6. **Score each clip** from 0 to 100:
   `Score = Σ wₖ · componentₖ`. The components are:
   - Hook: the best H in the first 3 s.
   - Cliffhanger: the best C in the last 5 s, weighted toward the final beat.
   - Energy: average intensity, minus a penalty for dead air.
   - Clarity: penalties for starting mid-sentence, heavy voice-over or scene jumps; a bonus when two or more characters clash.
   - Duration fit: a bell curve around 35 s.

   There is also a **turning-point prior**: an episode's own ending is written as a cliffhanger, so a clip that ends where the episode ends gets a minimum cliffhanger score.
7. **Select** the best non-overlapping clips per episode (greedy, with an overlap limit). Then **join** a strong hook segment with a later cliffhanger segment, from the same or nearby episodes, into multi-segment promos.
8. **Optional AI pass** (any provider). The model sees the transcript with line IDs plus the shortlist. It rates the hook, cliffhanger and stand-alone quality from 1 to 10, writes the text, and can move a cut, but **only by line ID**. Code re-computes timestamps and re-checks every rule.
   - final = (1 − μ)·rules + μ·AI.
   - The top promos are re-ordered with **pairwise** comparisons, asked in both orders.
9. **Rank.** Early-episode promos get a small bonus because they send new viewers to the start of the show. Promos that break a hard rule are kept out of the top list.

10. **Mood tags.** Each promo is tagged with up to two moods (Betrayal, Danger, Romance, Comedy, Horror…) from the stake categories in its lines, with the hook and cut lines counting double. They help pick the right audience for an ad.

11. **Learning from posted promos (optional).** Given a CSV of promos a team already posted, every candidate clip in those episodes becomes one row of about 50 features (the formula's components, what is in the hook line and the cut line, where the clip sits in the episode, silence, speakers, stake categories). A clip counts as "posted" if it starts within 3 s and ends within 5 s of a posted promo, so the model learns where promos start and stop rather than "longer clips overlap more" (an earlier labelling rule did exactly that and was replaced). Negatives are sampled 4 to 1. The model is a random forest, gradient-boosted trees (logistic loss, second-order leaf values, histogram splits, as in XGBoost) or logistic regression, written in plain Python so it also trains inside the browser in about 10 s. Its probability is blended with the formula: score = (1 − β)·formula + β·100·P(good). The type and β are chosen on validation episodes only.

### Why this design

| Idea | Source | Used for |
|---|---|---|
| Surprise vs. suspense | Ely, Frankel & Kamenica, *J. Political Economy* 2015; Wilmot & Keller, ACL 2020 | What a hook and a cliffhanger *are* |
| Turning points in screenplays, trailer moment selection | Papalampidi, Keller & Lapata, EMNLP 2019; arXiv 2021 | Episode-ending prior; keeping a human in the loop |
| Budgeted, non-redundant selection | Gygli, Grabner & Van Gool, CVPR 2015 | Choosing a diverse set; multi-segment promos within 60 s |
| Pairwise LLM ranking | Qin et al., Findings of NAACL 2024 | Ordering the top promos with any model |
