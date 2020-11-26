from itertools import chain, combinations
import json
from mmap import mmap
import numpy as np
from skimage.measure import label, regionprops
import os
from struct import pack, unpack


uint16 = 2  # bytes
uint32 = 4  # bytes


class BufferLocation:

    __slots__ = ('offset', 'length', 'buffer')

    def __init__(self, offset=0, length=0, buffer=None):

        self.offset = offset
        self.length = length

        if buffer:
            self.load(buffer)

    def load(self, buf):
        self.offset, self.length = unpack("II", buf)

    def dump(self):
        return pack("II", self.offset, self.length)

    def buf(self, mmap):
        """ Fetch buffer by location
        :param mmap:  (mmap) memory map to fetch from
        :return:      (buffer) byte buffer
        """
        return mmap[self.offset:self.offset + self.length]


class IIDFile:

    def __init__(self, fpath=None):
        """
        :param fpath:   (str) path to '.iid' file, creates empty IIDFile if None is provided.
        """

        self.exists = fpath is not None

        # Header
        if fpath:
            self.file = open(fpath, "r+b")
            self.mmap = mmap(self.file.fileno(), 0)
            self.filesize = os.stat(self.file.name).st_size

        self.header = Header(self)
        self.meta = Metadata(self)
        self.groups = Groups(self)
        self.lut = LookupTable(self)
        self.iids = IIDs(self)

        # Lazy
        self.segs = Segments(self)

    def dump(self):

        # Dump buffers
        segs = self.segs.dump()
        groups = self.groups._dump()
        meta = self.meta.dump()
        iids = self.iids.dump()
        lut = self.lut.dump()

        # Update buffer locations
        offset = Header.length

        self.lut.bufloc = BufferLocation(offset, len(lut))
        offset += len(lut)

        self.iids.bufloc = BufferLocation(offset, len(iids))
        offset += len(iids)

        self.meta.bufloc = BufferLocation(offset, len(meta))
        offset += len(meta)

        self.groups._bufloc = BufferLocation(offset, len(groups))
        offset += len(groups)

        self.segs.bufloc = BufferLocation(offset, len(segs))

        # Header
        self.header.update()
        header = self.header.dump()

        buffer = header + lut + iids + meta + groups + segs
        return buffer

    def save(self, fpath):
        with open(fpath, "w+b") as file:
            file.write(self.dump())

        self.file = open(fpath, "r+b")
        self.mmap = mmap(self.file.fileno(), 0)

    def add(self, address, domain, bbox, mask, group=None):
        """Add new IID segment

        :param address:  (bytes) iid name
        :param domain:   (bytes) domain name
        :param bbox:     (tuple) the mask bounding box (minr, minc, maxr, maxc)
        :param mask:     (numpy) binary buffer, size must match bounding box
        :param group:    (str) group name
        """

        # TODO: Assert that new IID isn't already in file.
        # In that case the IID's should either merge, or it should raise a warning.

        key = len(self.lut.entries)

        iid = IID(key=key, address=address, domain=domain)
        seg = Segment(key=key, bbox=bbox)
        seg.from_mask(mask, bbox)

        self.lut.add(key, iid, seg)

        if group:
            self.groups.add(name=group, keys=[key])

    def fetch(self, keys=None, all_keys=False, groups=None, iids=True, segs=True, everything=False):
        """Selective loading of entries from file

        :param keys:        (int|list) keys for entries to fetch.
        :param all_keys:    (bool) loads all entries. This differs from 'everything' in that it
                            still respects the groups, iids, segs arguments. Everything loads
                            the entire file.
        :param groups:      (list) fetch all entries in group
        :param iids:        (bool) fetch entry iid data
        :param segs:        (bool) fetch entry segment data
        :param everything:  (bool) load everything
        :return:            (list) of entries
        """

        if everything:
            keys = [entry.key for entry in self.lut.entries]
            iids = True
            segs = True

        else:
            if isinstance(keys, int):
                keys = [keys]
            if isinstance(groups, str):
                groups = [groups]

            if groups:
                keys = self.groups.get(groups, keys_only=True)
            elif all_keys:
                keys = [entry.key for entry in self.lut.entries]
            else:
                keys = [] if keys is None else keys

        keys = set(keys)

        if segs:
            self.segs.fetch(keys)
        if iids:
            self.iids.fetch(keys)

        self.lut.fetched_keys.update(keys)

        return [self.lut.entries[key] for key in keys]

    def look_for(self, addresses, domains=None, groups=None, segs=False):
        """Search for iids in file

        :param addresses:  (str|list) iid addresses to look for (query)
        :param domains:    (str|list) limit search to domains
        :param groups:     (str|list) limit search to groups
        :param segs:       (bool) fetch entry segment data
        :returns:          (list) of entries matching query
        """

        if not isinstance(addresses, list):
            addresses = [addresses]
        if groups and isinstance(groups, list) is False:
            groups = [groups]
        if domains and isinstance(domains, list) is False:
            domains = [domains]

        if groups:
            entries = self.groups.get(groups, iids=True, segs=False)
        else:
            entries = self.fetch(all_keys=True, iids=True, segs=False)

        if domains:
            entries = [entry for entry in entries if entry.iid.domain in domains]

        entries = [entry for entry in entries if entry.iid.address in addresses]

        if segs:
            # Only load the segments for entries matching query.
            entries = self.fetch([entry.key for entry in entries], segs=segs)

        return entries

    def filter(self, groups=None, area=None, domains=None, iids=True, segs=False):
        """Filter loaded entries in file

        :param groups:   (str|list) groups to include
        :param area:     (tuple) include segments within area range (min, max), where None ignores the bound,
                         ex.: (100, None) means that segments from 100px and upwards are returned.
        :param domains:  (bytes|list) include domains
        :param iids:     (bool) fetch entry iid data
        :param segs:     (bool) fetch entry segment data
        :return:         (list) of entries
        """

        iids = True if domains else iids  # Domain search requires iid data
        segs = True if area else segs  # Area search requires segment data

        if groups:
            if not isinstance(groups, list):
                groups = [groups]
            entries = self.fetch(groups=groups, iids=iids, segs=segs)
        else:
            entries = self.fetch(all_keys=True, iids=iids, segs=segs)

        if domains:
            if not isinstance(domains, list):
                domains = [domains]
            entries = [entry for entry in entries if entry.iid.domain in domains]

        if area:
            min_area = 0 if area[0] is None else area[0]
            max_area = float('inf') if area[1] is None else area[1]
            entries = [entry for entry in entries if min_area < entry.seg.area < max_area]

        return entries

    def at(self, x, y, only_loaded=False):
        """Get all segments at the given coordinate

        :param x:            (int) x coordinate
        :param y:            (int) y coordinate
        :param only_loaded:  (bool) only query loaded entries, otherwise everything will be fetched.
        :return:   (list) of entries
        """
        return self.region(bbox=(y-1, x-1, y+1, x+1), only_loaded=only_loaded)

    def region(self, bbox, only_loaded=False):
        """Get all segments within a given region

        :param bbox:         (tuple) search region (minr, minc, maxr, maxc).
        :param only_loaded:  (bool) only query loaded entries, otherwise everything will be fetched.
        :return:             (list) of entries
        """

        if only_loaded:
            entries = self.lut.fetched_entries()
        else:
            entries = self.fetch(everything=True)

        return [entry for entry in entries if entry.seg.intersects_bbox(bbox)]

    def overlap_graph(self, everything=True):
        """Computes a graph representation of segment overlaps.

        The graph nodes are the segments and the edges store the number of pixels (area)
        that overlap between them.

        :param everything:  (bool) fetch the entire file before searching the region,
                            otherwise only loaded segs will we queried.
        :return:            (tuple of lists) nodes, edges. Nodes are a list of
                            entries in the graph. Edges are tuples of two entries
                            and a value storing the number of pixels that overlap
                            between them. (A, B, pixels).
        """

        if everything:
            entries = self.fetch(everything=everything)
        else:
            entries = self.lut.fetched_entries()

        # Check all segment intersections once. Since there is no KD-tree or
        # other search optimization segment tests has to be computed for everything.
        # https://docs.python.org/3/library/itertools.html#itertools.combinations
        # Elements are treated as unique based on their position, not on their value.
        def _edge(a, b):
            return a, b, a.seg.intersects_segment(b)
        edges = [_edge(*pair) for pair in combinations(entries, 2)]
        edges = [edge for edge in edges if edge[2] > 0]

        return entries, edges


class Header:

    length = 12*uint32

    def __init__(self, iidfile):

        self.iidfile = iidfile

        if iidfile.exists:
            self.load()
        else:
            self.version = 0
            self.rformat = 0
            self.bufloc_lut = BufferLocation()
            self.bufloc_iids = BufferLocation()
            self.bufloc_meta = BufferLocation()
            self.bufloc_groups = BufferLocation()
            self.bufloc_segs = BufferLocation()

    def load(self):

        mmap = self.iidfile.mmap

        def _buf(offset, length):
            return mmap[offset*uint32: offset*uint32 + length*uint32]

        self.version = unpack("I", _buf(0, 1))[0]
        self.rformat = unpack("I", _buf(1, 1))[0]
        self.bufloc_lut = BufferLocation(buffer=_buf(2, 2))
        self.bufloc_iids = BufferLocation(buffer=_buf(4, 2))
        self.bufloc_meta = BufferLocation(buffer=_buf(6, 2))
        self.bufloc_groups = BufferLocation(buffer=_buf(8, 2))
        self.bufloc_segs = BufferLocation(buffer=_buf(10, 2))

    def dump(self):

        buffer = b''
        buffer += pack("I", self.version if self.version else 0)
        buffer += pack("I", self.rformat if self.version else 0)
        buffer += self.bufloc_lut.dump()
        buffer += self.bufloc_iids.dump()
        buffer += self.bufloc_meta.dump()
        buffer += self.bufloc_groups.dump()
        buffer += self.bufloc_segs.dump()

        return buffer

    def update(self):

        self.bufloc_lut = self.iidfile.lut.bufloc
        self.bufloc_iids = self.iidfile.iids.bufloc
        self.bufloc_meta = self.iidfile.meta.bufloc
        self.bufloc_groups = self.iidfile.groups._bufloc
        self.bufloc_segs = self.iidfile.segs.bufloc


class Metadata:

    def __init__(self, iidfile):

        self.iidfile = iidfile
        self.data = {}

        if iidfile.exists:
            self.bufloc = iidfile.header.bufloc_meta
            self.mmap = iidfile.mmap
            self.load()

    def load(self):
        o, l = self.bufloc.offset, self.bufloc.length
        self.data = json.loads(self.mmap[o:o+l].decode('utf-8'))

    def dump(self):
        return json.dumps(self.data).encode('utf-8')


class LookupTable:

    def __init__(self, iidfile, keys=None):
        """
        The order of the entries is immutable because the entry key corresponds to the index of the entries list.
        New entries must be appended at the end, and when removing entries, rather than deleting the object from
        the list the index should be set to None.

        :param iidfile:  (obj) IIDFile
        :param keys:     (list) keys to load, as when filtering by groups
        """

        self.iidfile = iidfile
        self.entries = []
        self.fetched_keys = set()  # Keys of entries with fetched segments

        if iidfile.exists:
            self.bufloc = iidfile.header.bufloc_lut
            self.mmap = iidfile.mmap
            self.load(keys)

    def add(self, key, iid, seg):
        entry = LookupTableEntry(key, iid, seg)
        self.entries.append(entry)
        return entry

    def fetched_entries(self):
        """
        :return:  (list) of fetched entries
        """
        return [self.entries[k] for k in self.fetched_keys]

    def load(self, keys=None):
        offset, length = self.bufloc.offset, self.bufloc.length
        num_entries = length // LookupTableEntry.length

        if keys:
            self.entries = [None] * num_entries  # Initiate full lut with None
            for key in keys:
                o = offset + key * LookupTableEntry.length
                self.entries[key] = LookupTableEntry(key=key, buffer=self.mmap[o:o + LookupTableEntry.length])
        else:
            for key in range(num_entries):
                o = offset + key * LookupTableEntry.length
                self.entries.append(LookupTableEntry(key=key, buffer=self.mmap[o:o + LookupTableEntry.length]))

    def dump(self):
        buffer = b''.join([entry.dump() for entry in self.entries])
        return buffer


class LookupTableEntry:

    __slots__ = ('key', 'iid', 'seg', 'buffer')

    length = 5*uint32

    def __init__(self, key, iid=None, seg=None, buffer=False):

        self.key = key
        self.iid = iid
        self.seg = seg

        if buffer:
            self.load(buffer)

    def load(self, buf):
        key, a, b, c, d = unpack("IIIII", buf)
        self.iid = IID(key=key, bufloc=BufferLocation(offset=a, length=b))
        self.seg = Segment(key=key, bufloc=BufferLocation(offset=c, length=d))

    def dump(self):
        return pack("I", self.iid.key) + self.iid.bufloc.dump() + self.seg.bufloc.dump()


class IIDs:

    def __init__(self, iidfile):

        self.iidfile = iidfile

        if iidfile.exists:
            self.bufloc = iidfile.header.bufloc_iids
            self.mmap = iidfile.mmap
            self.load()

    def load(self):
        offset = self.iidfile.header.bufloc_iids.offset
        for entry in self.iidfile.lut.entries:
            if entry:
                o, l = offset + entry.iid.bufloc.offset, entry.iid.bufloc.length
                buffer = self.mmap[o:o+l]
                entry.iid.load(buffer)

    def fetch(self, keys):

        offset = self.bufloc.offset
        iids = [self.iidfile.lut.entries[k].iid for k in keys]

        def buf(mmap, bufloc, offset=0):
            o, l = bufloc.offset, bufloc.length
            return mmap[offset+o:offset+o+l]

        [iid.load(buf=buf(self.mmap, iid.bufloc, offset)) for iid in iids]

    def dump(self):
        offset = 0
        buffer = []
        for entry in self.iidfile.lut.entries:
            buf = entry.iid.dump(offset=offset)
            buffer.append(buf)
            offset += len(buf)

        buffer = b''.join(buffer)
        return buffer


class IID:

    __slots__ = ('address', 'domain', 'key', 'bufloc')

    def __init__(self, address=None, domain=None, key=None, bufloc=None):
        """Individual IDentifier. The iid and domain values must be encoded
        byte strings, this is considered the base iid format. Decoding into
        int32, int64, str or other protocols are just interpretations of the
        bytes sequence. The interpretation is irrelevant as long as the
        iid is converted back to the original byte sequence before stored
        into another iid file. When storing an iid in an interpreted state,
        say outside an IIDFile, the protocol should be noted in order to be 
        able to convert back to the original byte sequence.

        :param address:  (bytes) byte string using struct.pack() or foo.encode()
        :param domain:   (bytes) byte string
        :param key:      (int) index position in file lookuptable. This argument
                         can be ignored when creating a new IID, since the
                         IIDFile.add() method will override the key field when
                         including the new IID object.
        :param bufloc:   (BufferLocation)
        """

        if address is not None and not isinstance(address, bytes):
            raise ValueError("'iid' must be encoded as bytes")

        if domain is not None and not isinstance(domain, bytes):
            raise ValueError("'domain' must be encoded as bytes")

        self.address = address
        self.domain = domain
        self.key = key
        self.bufloc = bufloc

    def load(self, buf):

        self.key, dom_len, iid_len = unpack("III", buf[:uint32*3])
        o = uint32*3
        self.domain = None if dom_len == 0 else buf[o:o+dom_len]
        o = o + dom_len
        self.address = buf[o:o+iid_len]

    def dump(self, offset=None):
        
        iid = self.address if self.address else b''
        dom = self.domain if self.domain else b''
        buf = pack("I", self.key) + pack("I", len(dom)) + pack("I", len(iid)) + dom + iid

        if offset is not None:
            self.bufloc = BufferLocation(offset, len(buf))

        return buf


class Groups:

    def __init__(self, iidfile):

        self._iidfile = iidfile
        self._entries = {}

        if iidfile.exists:
            self._bufloc = iidfile.header.bufloc_groups
            self._mmap = iidfile.mmap
            self._load()

    def _objects(self, groups):
        """Get the Group objects, which are hidden from the user.
        :param groups:  (list) of group names (str)
        :returns:       (list) group objects
        """
        return [self._entries[name] for name in groups]

    def _keys(self, groups):
        """Get the key set of the requested groups
        :param groups:  (list) of group names (str)
        :return:        (set) of keys in groups
        """

        if isinstance(groups, str):
            groups = [groups]

        # chain.from_iterable makes a list of list into a single list
        keys = chain.from_iterable([list(self._entries[group].keys_set) for group in groups])
        return list(set(keys))

    def _load(self):
        """Load groups from buffer"""

        o, l = self._bufloc.offset, self._bufloc.length
        buf = self._mmap[o:o + l]

        length, = unpack("I", buf[:uint32])
        groups = json.loads(buf[uint32:uint32+length].decode('utf-8'))

        o += uint32+length
        for group in groups:
            name, offset, length = group['name'], group['offset'] + o, group['length']
            self._entries[name] = Group(name, bufloc=BufferLocation(offset, length), mmap=self._mmap)

    def _dump(self):

        offset = 0
        buffer, groups = [], []
        for _, group in self._entries.items():
            buf = group.dump()
            buffer.append(buf)
            groups.append({'name': group.name, 'offset': offset, 'length': len(buf)})
            offset += len(buf)

        buffer = b''.join(buffer)
        groups = json.dumps(groups).encode('utf-8')

        return pack("I", len(groups)) + groups + buffer

    def add(self, name, keys=None):
        """Group a set of entries, will join keys if group exists.

        WARNING: entries, keys, iids or segs must already be part of the IID file

        :param name:     (str) group name
        :param keys:     (list) of LUT keys
        """

        try:
            group = self._entries[name]
        except KeyError:
            group = Group(name)

        if keys:
            group.add([self._iidfile.lut.entries[k] for k in keys])

        self._entries[name] = group

        return group

    def list(self):
        """
        :return:  (list) group names (str)
        """
        return sorted(self._entries.keys())

    def get(self, groups, keys_only=False, iids=True, segs=False):
        """Get entries in groups, entries will be fetched if not loaded

        :param groups:     (list) group names
        :param keys_only:  (bool) return keys instead of entries
        :param iids:       (bool) also fetch entry iid data
        :param segs:       (bool) also fetch entry segment data
        :return:           (list) lut entries
        """

        if isinstance(groups, str):
            groups = [groups]

        keys = set()
        for group in groups:
            group_keys = list(self._entries[group].keys_set)
            keys.update(group_keys)

        if keys_only:
            return keys
        else:
            return self._iidfile.fetch(keys=list(keys), iids=iids, segs=segs)


class Group:

    def __init__(self, name, bufloc=None, mmap=None):
        """A group is a key set that maps to entries in the lookup table.

        The group object should not be exposed directly to the user, all access to the group
        content should go through the IID file methods.

        :param name:     (str) name of group
        :param bufloc:   (bufloc) object buffer location
        :param mmap:     (mmap) iidfile buffer mmap (used to load on object creation)
        """

        self.name = name
        self.keys_set = None
        self.bufloc = bufloc

        if mmap:
            self.load(bufloc.buf(mmap))

    def add(self, entries):
        """Adds entries to group, maintains keys set.

        WARNING: only entries already loaded in the LUT of the current file can be added,
        adding entries from another IID file will add broken and bad keys to the group
        key set.

        :param entries:  (list) LUT entries
        """

        entries = set(entries)
        keys = [entry.key for entry in entries]
        if self.keys_set is None:
            self.keys_set = set(keys)
        else:
            self.keys_set.update(keys)

    def keys(self):
        """
        :returns:  (list) sorted keys
        """
        return sorted(list(self.keys_set))

    def load(self, buf, lut=None):
        """Loads group keys from buffer and maps group entries list to LUT if LUT is loaded.

        :param buf:  (buffer) group buffer
        :param lut:  (obj) IIDFile LUT
        """
        self.keys_set = set(unpack("%sI" % (len(buf) // uint32), buf))  # Load all keys from buffer

    def dump(self):
        """Dump object to bytes"""
        return pack("%sI" % len(self.keys_set), *self.keys_set)


class Segments:

    def __init__(self, iidfile):

        self.iidfile = iidfile

        if iidfile.exists:
            self.bufloc = iidfile.header.bufloc_segs
            self.mmap = iidfile.mmap

    def fetch(self, keys):

        offset = self.bufloc.offset
        segs = [self.iidfile.lut.entries[k].seg for k in keys]

        def buf(mmap, bufloc, offset=0):
            o, l = bufloc.offset, bufloc.length
            return mmap[offset+o:offset+o+l]

        [seg._load(buf=buf(self.mmap, seg.bufloc, offset)) for seg in segs]

    def dump(self):
        offset = 0
        buffer = []
        for entry in self.iidfile.lut.entries:
            buf = entry.seg._dump(offset=offset)
            buffer.append(buf)
            offset += len(buf)

        return b''.join(buffer)


class Segment:

    __slots__ = ('key', 'bbox', 'area', 'regions', 'bufloc')

    def __init__(self, key=None, bbox=None, area=None, regions=None, bufloc=None):
        """Segments represents the area covered by an IID. Segments are stored
        as regions, which are the disjoint areas that make up the segment mask.

        :param key:      (int) index in lookup table
        :param bbox:     (tuple) segment bounding box (minr, minc, maxr, maxc)
        :param area:     (int) area coverage in pixels
        :param regions:  (Regions)
        :param bufloc:   (BufferLocation)
        """

        self.key = key
        self.bbox = bbox
        self.area = area
        self.regions = regions
        self.bufloc = bufloc

    def _load(self, buf):
        s = uint32+uint16*4+uint32
        self.key, a, b, c, d, self.area = unpack("IHHHHI", buf[:s])
        self.bbox = (a, b, c, d)
        self.regions = Regions()
        self.regions._load(buf[s:])

    def _dump(self, offset=None):
        regs = self.regions._dump()
        buf = pack("IHHHHI", self.key, *self.bbox, self.area)
        buf += regs
        if offset is not None:
            self.bufloc = BufferLocation(offset, len(buf))
        return buf

    def bbox_xywh(self, center=False):
        """Bounding box as rectangle coordinates, with xy in upper-right corner.

        :param center:  (bool) place x,y at the center of the bounding box
        :return:        (tuple) x, y, w, h
        """
        return _bbox_to_xywh(self.bbox, center)

    def bbox_polygon(self):
        """Bounding box as polygon point coordinates, starting from upper-right corner.

        :return:  (list) of xy coordinate tuples
        """
        x, y, w, h = self.bbox_xywh()
        return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]

    def mask(self, bbox=None):
        """Get segment mask. The mask corresponds with the segment bounding box

        :param bbox:  (tuple) will crop mask to bounding box (minr, minc, maxr, maxc)
        :return:      (numpy) binary buffer
        """

        # Generate a mask from the segment regions.
        x, y, w, h = self.bbox_xywh()
        buf = np.zeros((h, w), dtype=np.bool)
        for reg in self.regions():
            minr, minc, maxr, maxc = reg.bbox
            minr = minr - y
            minc = minc - x
            maxr = maxr - y
            maxc = maxc - x
            buf[minr:maxr, minc:maxc] = buf[minr:maxr, minc:maxc] | reg.mask

        if bbox is None:
            return buf
        else:
            minr, minc, maxr, maxc = bbox
            x, y, _, _ = self.bbox_xywh()
            return buf[
                max(0, minr - y):maxr - y,  # Crop the mask by offsetting the bounding box
                max(0, minc - x):maxc - x   # to the mask local space.
            ]

    def from_mask(self, mask, bbox):
        """Sets the current segment mask. The mask must correspond to the bounding box.

        :param mask:  (numpy) binary buffer
        :param bbox:  (tuple) minr, minc, maxr, maxc
        """

        # Check that bounding box matches mask
        tmp = self.bbox
        self.bbox = bbox
        x, y, w, h = self.bbox_xywh()

        if mask.shape != (h, w):
            self.bbox = tmp
            raise ValueError("Mask shape must match bounding box")

        # Convert mask to regions
        regions = []
        area = 0
        for props in regionprops(label(mask, connectivity=2)):

            area = area + props.area
            bbox = props.bbox
            mask = props.image

            # Offset to segment bbox (places region bbox in world space)
            minr, minc, maxr, maxc = bbox
            minr = minr + y
            minc = minc + x
            maxr = maxr + y
            maxc = maxc + x

            regions.append(Region(mask=mask, bbox=(minr, minc, maxr, maxc)))

        self.area = area
        self.regions = Regions(regions=regions)

    def intersects_bbox(self, bbox):
        """Check if segment intersects with bounding bbox

        :param bbox:  (tuple) bounding box to intersect with (minr, minc, maxr, maxc)
        :returns:     (bool)
        """

        minr, minc, maxr, maxc = bbox
        xywh = _bbox_to_xywh(bbox, center=True)

        # Check if it intersects with the segment bounding box
        if not _bbox_intersects(self.bbox_xywh(center=True), xywh):
            return False

        # Check if it intersects with any of the region bounding boxes
        regions = [reg for reg in self.regions() if _bbox_intersects(reg.bbox_xywh(center=True), xywh)]
        if not regions:
            return False

        # Check if any region masks when cropped by the bounding box contain any True values,
        # if so the segment intersects with the bounding box.
        for reg in regions:
            x, y, _, _ = reg.bbox_xywh()
            if np.any(reg.mask[
                max(0, minr - y):maxr - y,  # Crop region mask by offsetting the bounding box
                max(0, minc - x):maxc - x   # to the region bounds.
            ]):
                return True

        # No intersections found
        return False

    def intersects_segment(self, entry):
        """Area of intersection with other segment

        :param entry:  (obj) will check if segments overlap, will override bbox argument if provided.
        :return:       (int) area of overlap in pixels
        """

        # Check if bounding box intersects
        if not _bbox_intersects(self.bbox_xywh(center=True), entry.seg.bbox_xywh(center=True)):
            return 0

        # Counts the number of intersecting pixels
        return int(np.sum(self.mask(bbox=entry.seg.bbox) & entry.seg.mask(bbox=self.bbox)))


class Regions:

    def __init__(self, regions=None):
        """Utility class for the regions of a segment.

        :param regions:  (list) of Region objects
        """
        self._entries = [] if regions is None else regions

    def __call__(self):
        """
        :return:  (list) of regions
        """
        return self._entries

    def _load(self, buf):
        o = 0
        while o < len(buf):
            length, = unpack("I", buf[o:o+uint32])

            reg = Region()
            reg._load(buf[o:o + length])
            self._entries.append(reg)

            o += length

    def _dump(self):
        return b''.join([reg._dump() for reg in self._entries])


class Region:

    __slots__ = ('mask', 'bbox')

    def __init__(self, mask=None, bbox=None):
        """Represents a region of a segment. Segments are broken into disjoint regions
        to compress the segment mask. In several cases where the segment is scattered
        and spread throughout the image, splitting these islands into independent buffers
        saves storing all the blank space in between.

        :param mask:  (numpy) binary array
        :param bbox:  (tuple) mask bounding box (minr, minc, maxr, maxc)
        """

        self.mask = mask
        self.bbox = bbox

    def _load(self, buf):

        # The mask buffer is padded to full bytes with np.packbits(), when unpacking
        # the resulting array must be clipped at 'num mask pixels', derived from the bbox.
        s = uint32 + uint16*4
        _, a, b, c, d = unpack("IHHHH", buf[:s])
        x = unpack("%sB" % (len(buf) - s), buf[s:])
        x = np.array(x, dtype=np.uint8)
        x = np.unpackbits(x).astype(np.bool)[:(c-a)*(d-b)]
        x = np.reshape(x, (c-a, d-b))

        self.bbox = (a, b, c, d)
        self.mask = x

    def _dump(self):
        x = np.ndarray.flatten(self.mask)
        x = np.packbits(x)

        buffer = b'' + pack("4H", *self.bbox) + pack("%sB" % len(x), *x)
        return b'' + pack("I", len(buffer) + 4) + buffer

    def bbox_xywh(self, center=False):
        """Bounding box as rectangle coordinates, with xy in upper-right corner.

        :param center:  (bool) place x,y at the center of the bounding box
        :return:        (tuple) x, y, w, h
        """
        return _bbox_to_xywh(self.bbox, center)

    def bbox_polygon(self):
        """Bounding box as polygon point coordinates, starting from upper-right corner.

        :return:  (list) of xy coordinate tuples
        """
        x, y, w, h = self.bbox_xywh()
        return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]


def _bbox_intersects(a, b):
    """Check if bounding boxes intersect
    https://gamedev.stackexchange.com/questions/586/what-is-the-fastest-way-to-work-out-2d-bounding-box-intersection
    :param a:  (tuple) xywh with xy centered
    :param b:  (tuple) xywh with xy centered
    :return:   (bool) does a intersect with b
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return abs(ax - bx) * 2 < (aw + bw) and abs(ay - by) * 2 < (ah + bh)


def _bbox_to_xywh(bbox, center=False):
    """Bounding box as rectangle coordinates, with xy in upper-right corner.

    :param center:  (bool) place x,y at the center of the bounding box
    :return:        (tuple) x, y, w, h
    """
    minr, minc, maxr, maxc = bbox
    x = minc
    y = minr
    w = maxc - minc
    h = maxr - minr
    if center:
        x, y = x + w / 2.0, y + h / 2.0
    return x, y, w, h
