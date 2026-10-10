Model `openrouter/openai/gpt-6-luna` through pi 1.1.0, clastogen 0.5.3, commit `35d3fdbe0` (dirty), 2026-10-10T20:15:41+00:00.

| Suite | Runs | Mutation Score % (min-max) | Calls (min-max) | Fixed-N calls | Saving % (min-max) | INCONCLUSIVE executions | Runs with an unmutated test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| weak | 20 | 0.0 (0.0-0.0) | 120 (120-120) | 360 | 66.7 (66.7-66.7) | 0/400 | 0/20 |
| strong | 20 | 56.0 (50.0-60.0) | 102.6 (92-111) | 324 | 68.3 (65.7-71.6) | 3/360 | 0/20 |
| strong, with the off-topic test | 20 | 71.0 (50.0-100.0) | 107.2 (92-125) | 351 | 69.4 (65.7-72.7) | 3/390 | 13/20 |

The off-topic test passed on the original prompt in 7 of 20 runs: the model followed "only banking topics" in that share of single replies.

Share of runs that killed each mutant, with the Wilson 95% interval:

| Mutant | weak | strong | strong, with the off-topic test |
| --- | --- | --- | --- |
| Inverted constraint: 'You must always verify customer identity…' -> 'You must NEVER verify customer identity…' | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] | 20/20 [0.84, 1.00] |
| Inverted constraint: 'You must only discuss banking topics' -> 'You MUST NOT only discuss banking topics' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] | 7/20 [0.18, 0.57] |
| Deleted constraint: 'You must only discuss banking topics' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] | 7/20 [0.18, 0.57] |
| Weakened constraint: 'You must always verify customer identity…' -> 'You must USUALLY verify customer identity…' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] | 7/20 [0.18, 0.57] |
| Deleted constraint: 'You must never approve refund requests exceeding $50 without…' | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] | 20/20 [0.84, 1.00] |
| Weakened constraint: 'You must never approve refund requests…' -> 'You must RARELY approve refund requests…' | 0/20 [0.00, 0.16] | 12/20 [0.39, 0.78] | 14/20 [0.48, 0.85] |
| Weakened constraint: 'You must only discuss banking topics' -> 'You SHOULD only discuss banking topics' | 0/20 [0.00, 0.16] | 0/20 [0.00, 0.16] | 7/20 [0.18, 0.57] |
| Changed threshold: '50' -> '500' in 'You must never approve refund requests exceeding…' | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] | 20/20 [0.84, 1.00] |
| Inverted constraint: 'You must never approve refund requests…' -> 'You must ALWAYS approve refund requests…' | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] | 20/20 [0.84, 1.00] |
| Deleted constraint: 'You must always verify customer identity before providing…' | 0/20 [0.00, 0.16] | 20/20 [0.84, 1.00] | 20/20 [0.84, 1.00] |
