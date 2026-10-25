## Findings

- **With the right groups, centring removes the topic effect.** At tau = 1 the twin gains +0.179 with true topics and
  +0.123 with content clusters; with no topic effect it costs about 0.01.
- **Score centring is the weaker tool.** Subtracting mean log-probabilities also removes the authors' own signal: it
  loses whenever the topic effect is weak (−0.132 at tau = 0) or the groups are small (−0.106 at two passages per topic).
- **Passages per topic decide the sign.** With content clusters the twin loses at one or two passages per topic
  (−0.047, −0.015) and gains from three on, up to +0.18 at 12.
- **A topic dominated by one author breaks the assumption.** When half a topic comes from one author, the gain drops
  to about 40% of the all-different case.
- **The cut matters most, and its safe side is the strict one.** A loose cut merges topics and loses; a strict cut
  groups little and costs nothing. With the right cut even two passages per topic gain (+0.066).
- **A small weight is the hedge.** At two passages per topic, weight 0.25 still gains +0.008 where weight 1 loses.

The same pattern shows on real out-of-fold predictions from Name That LLM (prompts cut to m answers, topics found by
content TF-IDF clustering): subtracting a quarter of every member's cluster-mean log-probability raised a three-member
blend from 0.626 to 0.644 with full prompts and gained at every m from 2 to 18 (+0.004 to +0.018), while heavier
centring gained with full prompts and lost with two or three answers per prompt.

Practical reading: before centring on a hidden test, look at the cluster sizes you get (singletons, pairs, 4 to 18)
and the purity the same cut gives on a labelled set with a similar structure; err towards a strict cut and a small
weight.

Source and tests: the `group-centring-bench` repo (MIT). Data: the attached dataset (CC0).
