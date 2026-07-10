# How AI Answers Faster: Speculative Decoding

Audience: English class final presentation, technical content section only.
Purpose: Provide the content part for two speakers; preparation process will be handled separately by other team members.
Duration target: about 4 minutes for these 6 slides.

## Slide 1: How AI Answers Faster
- Role: opening
- Speaker: Speaker 1
- Key message: We introduce speculative decoding as an inference acceleration method, not another quality-improvement method.

## Slide 2: The Problem: One Piece at a Time
- Role: problem framing
- Speaker: Speaker 1
- Key message: Normal language generation is token-by-token, so waiting time accumulates.

## Slide 3: The Core Idea
- Role: mechanism overview
- Speaker: Speaker 1
- Key message: A small model drafts, and the large model verifies; the large model remains in control.

## Slide 4: Speculative Decoding
- Role: concrete mechanism
- Speaker: Speaker 2
- Key message: The target model can accept several draft tokens at once and replace the first wrong token.

## Slide 5: Does It Hurt Quality?
- Role: quality guarantee
- Speaker: Speaker 2
- Key message: The draft model only proposes; the target model controls acceptance and final behavior.

## Slide 6: When It Works Best
- Role: conditions and limitations
- Speaker: Speaker 2
- Key message: Speedup depends on a fast draft model and reasonably accurate guesses.

## Prepared Q&A
Question: If the draft model sometimes makes wrong guesses, why can speculative decoding still speed up generation?
Answer: Because the target model verifies the draft. When the draft is good, multiple tokens can be accepted in one step. When the draft is wrong, the target model rejects the wrong part and continues normally. So the method works best when the draft model is both much faster and reasonably accurate.
