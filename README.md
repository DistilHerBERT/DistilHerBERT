# DistilHerBERT

## Datasets
### CCNet
I can’t find a ready-to-download dataset on the web. Instead, authors of the paper shared a [repository](https://github.com/facebookresearch/cc_net "repository") with tools for data extraction from Common Crawl (you have to apply code changes from [this pull request](https://github.com/facebookresearch/cc_net/pull/34 "this pull request") for it to work first; it's done on this branch already).

CommonCrawl is an org that releases a snapshot of web, notably in UTF-8 text format (WET).

According to papers with code the dataset is only in English and German.

Unclear:
- [ ] how to get the data
  * it seems the only choice is to run the pipeline / ask the authors whether they can send it / allegro team how they did it
how to get Polish version
- [X] what `head` means and how to get CCNet `head`
    * it means texts of highest quality (assessed by some score CCNet paper proposed)
- [ ] whether we have computational power to perform extraction they do in the paper
    * **Reconstructing the dataset by running our pipeline requires
a lot of resources and time.**
- [ ] what allegro team did for HerBERT
