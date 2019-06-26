+ Kristoffer
+ Omid
+ Tiago


# IID segmentation format (.iid)

> Still work in progress, use at your own risk {>>TODO: proper<<}

Lazy-loading format for search and retrival of image segmentation data. The iid-format is developed to efficiently search and retrive image segments labeled by Individual IDentifiers `IID`.

Individual identifiers are {==universal==}{>>Find proper term<<} names for object, concept or event as they occure throughout a data-body. Similar to `URI`, `URL` or multi-hash `IPFS`, yet without a prescribed structure.

An iid-file stores a segmentation map of an image file. The map tells which part of the image corresponds to which `IID`. 

Memory-mapping and lazy-loading enables peaking into the iid-file without parsing the entire buffer. Selective parsing is usefull when a single image contains hundres of thousands of segments.

> Lazy-loading is intended to assist fast search. For making change to an iidfile load the full file since only data that is loaded will be saved back to disk. This means that a partially loaded file will result in data-loss if you overwrite the existing file.

Features
--------
The iidfile contains a lookup table that maps the file location of each IID and it's corresponding segment.

The python parser uses the python `mmap` utility to lazy-load data as needed. When opening a file only the header and the lookuptable is loaded. The actual IID and segments are only loaded if needed.

The format is designed for several types of queries:

+ image by IID
+ IID by image
+ segment by IID
+ IID by segment
+ segements by image section

Segments and regions
--------------------

Segment are composed of one or more region. A region is defined by a bounding box and a binary mask such that it masks a section of the image. The structure is similar to that of {>>TODO<<} skimage-region. The segment also stores a bounding box encompasing all regions and an area attribute describing the number of pixels in the segment.

![segmentation example](path)

Segments can overlap, multi-label classification, so the same parts of an image can be labled by multiple segments. The formats supports up to `2^32-1` segments with a matching `IID`s.

Groups
------

Groups are used to query subsets of the segments. Subsets can be segments that share spesific properties, such as segments mapping _representation classes_. Querying a subset is more effective. As in the case of the _representation classes_ that might be less than 10 iids in a iidfile that contain the hundreds of thousands of other segments.

Groups can also be a useful way distinguish between different classes of short iids (few bytes). Shorter iids saves memory when dealing with ephemeral or local individuals, but they are prone to collision which groups can help mitigate.

Metadata
--------

Metadata is stored in a generic json format, it's open but follows a naming convension.

__Naming convensions__

```
{
  "image": {
    "height": <pixels>    image height
    "width": <pixels>     image width
  },
  "camera": {             camera properties

  }
}
```


Usage examples
--------------

```
iidfile = IIDFile(fpath='somefile.iid')
iidfile.fetch(everything=True)
```

```
iidfile = IIDFile(fpath='somefile.iid')
iidfile.fetch(groups='iid', iids=True)  # Only load iids in group 'iid', this will NOT load the segments.
```

```
iidfile = IIDFile(fpath='somefile.iid')
iidfile.fetch()
```


File format
-----------

Described with psuedo-BNF. Types contained in brackets `{ }` represents lists.

##### Common types

```
char      ::= 1 byte
uint8     ::= 1 byte
uint16    ::= 2 byte                       short
uint32    ::= 4 byte                       32-bit unsigned integer
len       ::= uint32                       length in bytes
string    ::= len { char }
bufloc    ::= uint32 uint32                buffer location, offset and length
```

##### File structure

```
file      ::= header
              lut                          lookup table from key to iid and segment
              iids                         
              meta                         meta data as json
              groups                       groups
              segments                     segments

header    ::= version                      format version number
              rformat                      TODO: resource format (image, text, image sequence, ...)
              bufloc_lut                   location of lookuptable
              bufloc_iids                  location of iid data (relative)
              bufloc_meta                  location of meta data
              bufloc_groups                location of groups
              bufloc_segs                  location of segments

version   ::= uint32
rformat   ::= uint32

lut       ::= { key iid seg }               
iids      ::= { key len len bytes bytes }  key, len, len, domain, iid
meta      ::= string                       json
                                           
groups    ::= len json { group }           len, group list, group data
group     ::= { key }                      list of LUT keys

key       ::= uint32                       lookup key, maps to iid and segment
iid       ::= bufloc                       location of iid
seg       ::= bufloc                       location of segment

segments  ::= { key bbox area { region } }
region    ::= len bbox { byte }            len, bbox, mask buffer

area      ::= uint32                       counted in pixels
bbox      ::= uint16*4                     minr, minc, maxr, maxc
```