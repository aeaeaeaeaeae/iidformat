# IID fileformat (.iid)

> DISCLAIMER: This is still under development, it's been implemented to store the segmentation data for [ae73edb74571e4e2](https://www.instagram.com/ae73edb74571e4e2) so it works but changes will happen.

`.iid` is a memory-mapped format for archival, search and retrival of image segmentations. This repository contains a python implementation and some example files.

![segmentation](https://github.com/aeaeaeaeaeae/data/blob/master/segmentation.jpg)

#### Segments and Individual-IDentifier

An _IID-file_ only stores the segmentation, not the actual image. The segments are binary masks that covers sections of the image. The segments are labeled with **Individual IDentifiers** `IID`. These are _arbritary_ and _globally unique_ IDs associated with a spesific _individual_, where an individual is a distinct entity (object, concept, event and so on). 

An IID is _arbritary_ in the sense that it does not have any formal relationship with the data it labels (it's not a hash). It's _globally unique_ in the sense that it is a global label, any data labeled with a specific IID is considered part of this individual.

IIDs lets the data drive its own the definition. Since an IID is arbritary (made from random bytes) it is also meaningless, meaning therefore flows from the data to the label. The data (as experience) defines the individual (as word).

An IID is composed of two raw byte-strings, _domain_ and _iid_. The _domain_ defines the context (what language/protocol) and _iid_ defines the name of that individual in that context. There is no fixed length for either the _domain_ or _iid_ bytestring.

Features 
---------------

### Memory-mapped

An IID-file is structured with a header and lookup table that maps the memory location of the IIDs and the corresponding segments. This enables selective (lazy) loading of file-content. Lazy-loading is usefull when searching through large IID-files since you can limit parsing to the parts needed. This python implementation uses `mmap` to parse spesific parts of the file-buffer.

### Segments and regions

Every IID has a corresponding segment. The segment are composed of one or more region. A region is defined by a bounding box and a binary mask. The structure is inspired by [region_props](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops) from [skimage](https://scikit-image.org/). Segments can overlap, supporting multilabeling of the same pixels. The current implementation supports up to `2^32-1` segments.

> Regions saves us from large but empty buffers when segment pixels are scattered across the image.

__Segment attributes__

+ __area__: number of masked pixels
+ __bbox__: segment bounding box
+ __regions__:
    - bbox: region bounding box
    - buffer: binary byte buffer

![image segment regions](https://github.com/aeaeaeaeaeae/data/blob/master/image_segments_regions.jpg)

### Groups

Groups are sets of IIDs (and segments). It's an optimization feature enabling selective querying and loading of IIDs.

Groups lets you distinguish IIDs with particular properties. F.ex. a set of segments that labels classes of objects (trees, bushes, stones, grass) can be differentiated from segments of individuals (tree A, tree B, stone X, stone Y, ...). There might be tens of thousands of individuals yet only a few classes, groups lets you limit your loading and search to class IIDs.

### Metadata

Embedded JSON storing additional data, such as camera parameters and image properties.

__Naming convension__

```json
{
    "image": {
        "width": "integer (pixels)",
        "height": "integer (pixels)"
    }, 
    "camera": {
        "translate": "float 3 vector", 
        "rotate": "float 3 vector (degrees)", 
        "fstop": "float", 
        "focus": "float"
    }, 
    "keyframes": {
        "frame": "integer", 
        "lastFrame": "integer", 
        "firstFrame": "integer"
    }
}
```

Usage
-----

Clone repository and link it up:

```python
import sys
sys.path.insert(0, path_to_repo)

from iidfile import IIDFile
```

### Example files

📦 [download link]()

### Example code

__Creating IID-file__

```python
# Creates empty file
file = IIDFile()

# Populate file
for s in segments:

    iid = IID(s.iid, s.domain)
    regs = Regions([Region(r.buffer, r.bbox) for r in s.regions])
    seg = Segment(regions=regs, bbox=s.bbox, s=segment.area)

    file.add(iid=iid, seg=seg, group='somegroup')

file.save(fpath)
```

__Loading IID-file__

```python
# Loads entire file
iidfile = IIDFile(fpath='somefile.iid')
entries = iidfile.fetch(everything=True)

# Only load IIDs in group 'cls', this will NOT load the segments.
iidfile = IIDFile(fpath='somefile.iid')
entries = iidfile.fetch(groups=['cls'], iids=True)

# Fetches both segments and iids of group 'cls'
iidfile = IIDFile(fpath='somefile.iid')
entries = iidfile.fetch(groups=['cls'], iids=True, segs=True)

# Fetches first 50 iids
iidfile = IIDFile(fpath='somefile.iid')
entries = iidfile.fetch(keys=range(50), iids=True)
```

Fileformat
----------

Described with psuedo-BNF. Types contained in brackets `{ }` represents lists.

##### Common types

```
char      ::= 1 byte
uint8     ::= 1 byte
uint16    ::= 2 byte                       short
uint32    ::= 4 byte                       32-bit unsigned integer
len       ::= uint32                       length in bytes
string    ::= len { char }
json      ::= string                       raw json string
bufloc    ::= uint32 uint32                buffer location, offset and length
```

##### File structure

```
file      ::= header
              lut                          lookup table (LUT) from key to iid and segment
              iids                         
              meta                         meta data as json
              groups                       groups
              segments                     segments

header    ::= version                      format version number
              rformat                      TODO: consider removing?
              bufloc_lut                   location of lookuptable
              bufloc_iids                  location of iid data (relative)
              bufloc_meta                  location of meta data
              bufloc_groups                location of groups
              bufloc_segs                  location of segments

version   ::= uint32
rformat   ::= uint32

lut       ::= { key iid seg }               
iids      ::= { key len len bytes bytes }  key, len, len, domain, iid
meta      ::= json                         metadata and other info
                                           
groups    ::= len json { group }           len, group list, group data
group     ::= { key }                      list of LUT keys

key       ::= uint32                       LUT key, maps to iid and segment
iid       ::= bufloc                       location of iid
seg       ::= bufloc                       location of segment

segments  ::= { key bbox area { region } }
region    ::= len bbox { byte }            len, bbox, mask buffer

area      ::= uint32                       counted in pixels
bbox      ::= uint16*4                     minr, minc, maxr, maxc
```
