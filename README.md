IID fileformat (.iid)
=====================

`.iid` files contains a list of masks and labels associated with an image. It is a file format for storing image segmentations mapped to **Individual IDentifiers** (IIDs). 

This package provides methods for loading, saving, searching and computing data from `.iid` files. 

> Why? Because it was needed to map this [ae73edb74571e4e2](https://www.instagram.com/ae73edb74571e4e2).

## Installation

```bash 
pip install iidfile
```

## Quick Example

~~~python
from iidfile import IIDFile

mask = np.ones(shape, dtype=np.bool)
bbox = (minr, minc, maxr, maxc)

# Create iidfile
iidfile = IIDFile()
iidfile.add(address=b'foo', domain=b'bar', bbox=bbox, mask=mask)
iidfile.save('path/file.iid')

# Load iidfile
iidfile = IIDFile(fpath="path/file.iid")
entries = iidfile.fetch(everything=True)
~~~

## Overview

IID files are not image files, rather they store labeled segments that maps a image file. The IID files have no set resolution, instead every segments has a bounding box that places the segment mask on the image it maps. There is no limit to the number segments or the ways that they can overlap. Overlaps has no depth hierachy, there is no concept of in front or behind.

### Entries

Internally an `.iid` file contain a lookup table, where each _entry_ stores an IID and a segment pair. Each entry have a key, which correspond to their position in the lookup table.

```python
# Add entry (iid, segment) to iidfile
iidfile.add(address=b'foo', domain=b'bar', bbox=bbox, mask=mask)

# Get the first entry
entry = iidfile.fetch(everything=True)[0]

# Attributes
key = entry.key
address = entry.iid.address
domain = entry.iid.domain
mask = entry.seg.mask()
bbox = entry.seg.bbox
```

### Segments

Segments are binary masks placed by a bounding box. These masks are paired to a single IID each, and thereby maps the part of the image they cover to that spesific IID.

### IID (Individual IDentifiers)

IIDs are labels composed of an _address_ and a _domain_. Both the domain and address is stored as raw byte strings. There is no length limit to this bytestring. The domain defines the context (what language/protocol) and address defines the name of the label in that context.

The purpose of IIDs is to provide globally unique labels that can link segments across `.iid` files.

```python
# Masks are binary numpy buffers
mask = np.zeros(shape, dtype=np.bool)

# Bounding boxes determine the placement of the mask in the image coordinate space
h, w = mask.shape
bbox = y, x, y+h, x+w  # minr, minc, maxr, maxc

iidfile.add(address=b'foo', domain=b'bar', bbox=bbox, mask=mask)
```

## Features

An IID-file is structured with a lookup table that maps the memory location of the entry data. This enables selective loading of file-content. Selective loading can limit the parsing needed when searching through large `.iid` files. This python implementation uses `mmap` to parse spesific parts of the file-buffer.

#### Groups

Groups are sets of entries, they enable faster and targeted loading of segments.

```python
# Adding entry to group on creation
iidfile.add(address=b'foo', domain=b'bar', bbox=bbox, mask=mask, group='sys')
iidfile.save('path/file.iid')

# Only load segments in specified group
iidfile = IIDFile(fpath='path/file.iid')
entries = iidfile.fetch(groups=['sys'])
```

#### Metadata

An embedded dict stores any additional meta data, this can be camera parameters or info about the external image. The dict must be JSON serializable.

```python
metadata = iidfile.meta.data
```

#### Segments and regions

Internally segments are split up into smaller regions of connected patches, this compresses the segment mask buffer, especially when the mask is large and scattered.

![image segment regions](https://github.com/aeaeaeaeaeae/data/blob/master/image_segments_regions.jpg)

## References

> Use docstring for details.

#### IIDFile

+ `add` - Add an entry (iid and segment) to file
+ `save` - Save file to disk
+ `fetch` - Selective loading of entries
+ `look_for` - Search for iids
+ `filter` - Filter loaded entries
+ `at` - Get segments intersecting at position
+ `region` - Get segments intersecting with a given bounding box
+ `compute_overlap` - Computes an overlap graph from loaded segments


## Examples

📦 [Download some example .iid files]()

#### Assign segments and save to disk

```python
from iidfile import IIDFile

iidfile = IIDFile()  # Creates empty iidfile

data = [
  (b'tree', numpy_buffer_1, xy_1),
  (b'leaf', numpy_buffer_2, xy_2),
  (b'root', numpy_buffer_3, xy_3)
]

for iid, mask, xy in data:

  # Format bbox to minr, minc, maxr, maxc
  x, y, w, h = *xy, mask.shape[1], mask.shape[0]
  bbox = (y, x, y+h, x+w)

  iidfile.add(address=iid, domain=b'example', bbox=bbox, mask=mask)

iidfile.save(fpath='example.iid')
```

#### Fetch content from saved file

```python
# Load everything
iidfile = IIDFile(fpath='example.iid')
entries = iidfile.fetch(everything=True)

# Only load entries in group 'sys', and only load their iids,
# this will not load their segments.
iidfile = IIDFile(fpath='example.iid')
entries = iidfile.fetch(groups=['sys'], iids=True)

# Load all entries and only load their segments, 
# this will not load their iids.
iidfile = IIDFile(fpath='example.iid')
entries = iidfile.fetch(all_keys=True, segs=True)

# Only load the first 50 entries.
iidfile = IIDFile(fpath='example.iid')
entries = iidfile.fetch(keys=range(50), iids=True)
```

#### Query, filter and look for content

```python
# Finds all segments intersecting at x:400, y:600.
iidfile = IIDFile(fpath='example.iid')
entries = iidfile.at(400, 600)

# Only fetch entries in the group 'sys', do not load thier
# iids and find which intersects with the bounding box.
iidfile = IIDFile(fpath='example.iid')
iidfile.fetch(group=['sys'], iids=False)
entries = iidfile.region(bbox, only_loaded=True)

# Check if iid addressess exists in file.
iidfile = IIDFile(fpath='example.iid')
entries = iidfile.look_for(addresses=[b'tree', b'leaf'])

# Filter entries in file, this return entries in the group
# 'rep' with a segment area of 4000 and greater.
iidfile = IIDFile(fpath='example.iid')
iidfile.fetch(everything=True)
entries = iidfile.filter(groups=['rep'], area=(4000, None))
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
