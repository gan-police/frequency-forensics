<!--
<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>
-->

# frequency-detection

Hi guys, welcome!
The first step will be to reproduce some of the results from
https://papers.nips.cc/paper/2020/file/1f8d87e1161af68b81bace188a1ec624-Paper.pdf
I will start to do that next week.

## whats the plan?
 - I think it may be a good idea to use the fwt instead of the fft to find GAN
   generated conted, because of Gibbs phenomenon:
   https://en.wikipedia.org/wiki/Gibbs_phenomenon
 - Faces have sharp edges i.e. at the forehead - hair transition. I expect
   this to happen with the method above.
 => See if we can find a wavelet that is better suited to GAN-content 
    dectection and try to write a paper about it.

## GAN Review List

https://docs.google.com/spreadsheets/d/1uWQBdcbQIOcomveN912X8m3U0W_wORY1ZF5tMvGmmSA/edit?usp=sharing

## Motivation List
Distopian future where you dont know what's real and what's not. Thus, we have to prepare beforehand
for this race by having machines that are able to recognize the difference between fake and real content.

For example:
- People already started using GANs to fabricate fake content (e.g., images, audios, etc.).
  - Example: Biden's video
  - Example: Homer's voice
- There are existing fake versions of Copyright work that is being. 
  - Youtube videos that have been copied/modified (e.g., 2x speed video and you want to recognize)
  - Recognizing real paintings 

## Goals
- Flag such content so that people is aware that the image/audio is fake.
- Having an open source software available for the community (e.g., media) to check the vericity of content.
- Benchmark for the community and extensions (e.g., paper comes and they have to be able to avoid being recognized
by our software but at the same time new comes).

