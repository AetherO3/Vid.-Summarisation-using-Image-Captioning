# Vid.-Summarisation-using-Image-Captioning

Making Video Summarisation tool using Image Captioning as it's core to allow for more modularity.


The idea is to use image captioning a few frames(say every 12th frame in a 24 fps film, so every .5 sec), to generate a small caption i.e. description of the 
image and stack these descriptions for a set time period(say 10 mins) to then generate a more general summary for that time duration, adjustable as per the user 
requirements and the goal is to see if segmentation of the tasks allows for smaller and less computation demanding models with similar accuracies to state of the 
art video summarisers.

The methodology used is a combination of VGG16 for feature extraction from the frames and Transformers for caption generation as well as summarisation. 


Application scope is primarity in surveillance eg. a house security camera or a praking lot camera, something that does not need full attention by the user.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

A variant of this can be using the captions for the time sequence to then detect anomalies between different frames for more security based applications like intrusion detection.
