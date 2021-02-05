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

# whats the plan?
 - I think it may be a good idea to use the fwt instead of the fft to find GAN
   generated conted, because of Gibbs phenomenon:
   https://en.wikipedia.org/wiki/Gibbs_phenomenon
 - Faces have sharp edges i.e. at the forehead - hair transition. I expect
   this to happen with the method above.
 => See if we can find a wavelet that is better suited to GAN-content 
    dectection and try to write a paper about it.

