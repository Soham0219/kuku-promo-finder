# Key decisions (for the interview)

Know each decision, the reason for it, and what you'd change.

1. **Script-only, rules first, AI optional.** No video was provided and there are no labelled promos. Rules are free, instant and explainable, and they can't invent timestamps. The AI is used only where meaning matters (a silent proposal, a verdict) and only on a shortlist, which keeps the cost at about $0.30 per show (about $0.60 after Google's listed 1 Jan 2027 price rise).
2. **Hook = surprise, cliffhanger = suspense.** This follows Ely, Frankel & Kamenica (2015). Surprise is a big jump in what the viewer believes; suspense is uncertainty about what comes next. It gives a principled reason for each signal instead of an arbitrary list.
3. **Noisy-OR for stakes.** One shock category already makes a moment high-stakes, and more categories add with diminishing returns. A plain sum would let a line packed with keywords score above 1.
4. **Exhaustive search, not sampling.** About 300 distinct candidate clips per episode (109–664) is tiny, so every clip is scored and the best one is guaranteed to be found.
5. **Cuts snap to camera shots and never land inside a line.** A promo that cuts mid-word is unusable, so this is a hard rule with a test.
6. **Turning-point prior.** Microdrama episodes are written to end on a cliffhanger, so a clip that ends where the episode ends gets a minimum cliffhanger score. It is a general genre property, not a rule for this show. Its measured effect is small: removing it drops held-out recall from 0.67 to 0.62 (one pick). Endings that need meaning to understand (the proposal, the verdict) are still missed.
7. **Tuned on episodes 1–25, tested on 26–50.** Held-out results: recall 0.67 (16 of 24 picks), top-1 0.62. Tuning lifted the tuning half (0.71 → 0.76) but moved the held-out half by only one pick (17 → 16), so the weights are not overfitted, but tuning buys little.
7b. **Word lists must not contain show phrases.** The recheck found phrases copied from lines in these scripts in the word lists. They added 3 held-out finds (0.79 before, 0.67 after removal), so the earlier score was inflated. Only general drama words remain.
8. **The AI may only refer to line IDs.** Code converts IDs to timestamps and re-checks every rule. Invented IDs, out-of-range scores and API failures fall back to the rules result (all covered by tests).
9. **Pairwise ranking for the top list.** "Is A or B the better promo?", asked in both orders, is more reliable than asking for scores out of 10 (Qin et al., Findings of NAACL 2024), and it works with small or free models.
10. **Genre mode, without touching the tuned drama scoring.** Promos are cut differently per genre: a comedy cuts on an awkward beat, a thriller on a threat or a countdown. Genres are config profiles that only *add to* the base settings, and drama (the input pack's genre) has none, so the measured results above still hold. The input pack has one genre, so the other profiles are tested on made-up demo shows only; their weights are not tuned.
11. **Learn from labels, but only claim what the test shows.** A learned model (random forest, boosted trees or logistic regression) can replace the hand-set weights once a team has a posting history. Chosen on validation episodes only, it found 16 of 24 held-out picks, the same as the formula; boosted trees found 21, but picking it after seeing the test would be cheating. 45 labels are too few to show a gain; Kuku's real posting history would not be.
12. **Flag, don't hide.** Sexual content, weak hooks, weak endings and multi-segment joins are flagged for the editor. The tool proposes; the editor decides.

**What I'd do next:**
- Run the AI reviewer with Kuku's model.
- Have a real editor re-label the 45 picks.
- Learn the weights from editors' accept/reject decisions.
- Add audio and video signals once video is available.
- A/B test promos on 3-second hold rate, click-through and episode-1 starts.
