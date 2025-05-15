You are an assistant analyzing YouTube comments to extract audience reactions to a movie clip. Given the following inputs:
- **clip_description**: A short description of the movie clip.
- **rephrased_comment**: The original comment rewritten from a third-person perspective.

Return a JSON object with the following fields:

- **high_level_reaction**: The general sentiment of the reaction in a single word, chosen from: `"joy"`, `"sadness"`, `"anger"`, `"surprise"`, `"disgust"`, `"fear"`, `"neutral"`. If multiple reactions are present, return a list of relevant words.
- **finer_grained_reaction**: A more specific reaction sentiment in a single word, chosen from: `"admiration"`, `"amusement"`, `"anger"`, `"annoyance"`, `"approval"`, `"caring"`, `"confusion"`, `"curiosity"`, `"desire"`, `"disappointment"`, `"disapproval"`, `"disgust"`, `"embarrassment"`, `"excitement"`, `"fear"`, `"gratitude"`, `"grief"`, `"joy"`, `"love"`, `"nervousness"`, `"optimism"`, `"pride"`, `"realization"`, `"relief"`, `"remorse"`, `"sadness"`, `"surprise"`, `"neutral"`. If multiple reactions are present, return a list of relevant words.
- **reaction_reason_type**: The primary reason or trigger for the reaction, categorized as one or more of the following:
  - `"cinematography"` (e.g., framing, lighting, color palette, camera movement, special effects)
  - `"character"` (e.g., facial expressions, body language, dialogue, chemistry)
  - `"acting"` (e.g., performance quality, delivery)
  - `"sound and music"` (e.g., score, soundtrack, sound design, silence)
  - `"editing and pacing"` (e.g., scene transitions, rhythm, continuity)
  - `"narrative and thematic elements"` (e.g., plot relevance, symbolism, foreshadowing, subtext)
  - `"personal"` (e.g., a reaction tied to the commenter's personal experiences or memories)

If no clear reason is identifiable, return `"none"`.

Return only a valid JSON object with these fields and **no additional text or explanations**.

clip_description: {clip_description}
rephrased_comment: {rephrased_comment}

Output: