# Group Centring When Test Topics Must Be Found

Passages that answer the same prompt share a topic effect in a classifier's features. Subtracting each group's mean
and reading what is left with a second head trained the same way (the centred "twin") removes that effect. In the
Name That LLM competition the top public notebooks do this on content clusters of the hidden test, because the test
has no prompt ids.

This notebook measures, on a simulation where the truth is known, how much centring gains and what it costs when the
groups have to be found by clustering. 18 authors write passages on many topics; each passage has style features
(author plus a shared topic effect plus noise) and content features (topic plus noise) used only to find the groups.
Linear discriminant heads are fitted on a balanced training batch (150 topics, every author once per topic).

Rules compared, as gains in balanced accuracy over the plain head:

- `twin_true`: plain plus centred head, with the true topics as groups (the ceiling).
- `twin_est`: the same with content clusters (cosine distance, average linkage, cut 0.85).
- `score_centre_est`: no second head; subtract each cluster's mean log-probability.

Everything runs on CPU in a few minutes. The package source is embedded in the next cell.
