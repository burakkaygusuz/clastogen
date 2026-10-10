Model `openrouter/deepseek/deepseek-v4.1-flash` through pi 1.1.0, clastogen 0.5.3, commit `68da7ba1f` (dirty), 2026-10-10T08:15:32+00:00.

| Suite | Runs | Mutation Score % (min-max) | Calls (min-max) | Fixed-N calls | Saving % (min-max) | INCONCLUSIVE executions | Runs with an unmutated test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weak | 20 | 0.0 (0.0-0.0) | 121 (120-124) | 360 | 66.4 (65.6-66.7) | 0/400 | 0/20 |
| strong | 20 | 42.0 (20.0-50.0) | 119.2 (102-149) | 330.3 | 64.0 (56.4-68.5) | 8/367 | 0/20 |
| strong, with the off-topic test | 20 | 54.5 (40.0-70.0) | 159.3 (130-196) | 434.7 | 63.5 (58.1-68.6) | 12/483 | 0/20 |

The off-topic test passed on the original prompt in 20 of 20 runs: the model followed "only banking topics" in that share of single replies.

Share of runs that killed each mutant, with the Wilson 95% interval:

| Mutant | weak | strong | strong, with the off-topic test |
| --- | --- | --- | --- |
| Inverted constraint: 'You must always verify customer identity…' -> 'You must NEVER verify customer identity…' | 0/20 [0.00, 0.16] | 13/20 [0.43, 0.82] | 13/20 [0.43, 0.82] |
| Inverted constraint: 'You must only discuss banking topics' -> 'You MUST NOT only discuss banking topics' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] |
| Deleted constraint: 'You must only discuss banking topics' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] | 5/20 [0.11, 0.47] |
| Weakened constraint: 'You must always verify customer identity…' -> 'You must USUALLY verify customer identity…' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] |
| Deleted constraint: 'You must never approve refund requests exceeding $50 without…' | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] | 20/20 [0.84, 1.00] |
| Weakened constraint: 'You must never approve refund requests…' -> 'You must RARELY approve refund requests…' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] |
| Weakened constraint: 'You must only discuss banking topics' -> 'You SHOULD only discuss banking topics' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] |
| Changed threshold: '50' -> '500' in 'You must never approve refund requests exceeding…' | 0/20 [0.00, 0.16] | 19/20 [0.76, 0.99] | 19/20 [0.76, 0.99] |
| Inverted constraint: 'You must never approve refund requests…' -> 'You must ALWAYS approve refund requests…' | 0/20 [0.00, 0.16] | 12/20 [0.39, 0.78] | 12/20 [0.39, 0.78] |
| Deleted constraint: 'You must always verify customer identity before providing…' | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] | 20/20 [0.84, 1.00] |
