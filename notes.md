
Since we use a gpu, it might be possible to use softmax itself instead of hierarchical softmax, since it is likely quite fast too.

Unclear parts of the paper:
- What happens when the phrase is shorter than the window size + 1? Do we simply ignore phrases longer than it? This discards some training instances, so we pad the first words with 0s.
