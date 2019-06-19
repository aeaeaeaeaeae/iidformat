from itertools import chain
import json
from mmap import mmap
import numpy as np
import os
from struct import pack, unpack


uint16 = 2  # bytes
uint32 = 4  # bytes


class BufferLocation:

    def __init__(self, offset=0, length=0, buffer=None):

        self.offset = offset
        self.length = length

        if buffer:
            self.load(buffer)

    def load(self, buf):
        self.offset, self.length = unpack("II", buf)

    def dump(self):
        return pack("II", self.offset, self.length)


def dump_str(s):
    return bytes(s) if s else b''


class IIDFile:

    def __init__(self, fpath=None, find=None):

        self.exists = fpath is not None

        # Header
        if fpath:
            self.file = open(fpath, "r+b")
            self.mmap = mmap(self.file.fileno(), 0)
            self.filesize = os.stat(self.file.name).st_size

        self.header = Header(self)

        if find:
            self.lut = LookupTable(self)
            self.iids = IIDs(self)
            self.meta = Metadata(self)
            self.groups = Groups(self)
        else:
            self.lut = LookupTable(self)
            self.iids = IIDs(self)
            self.meta = Metadata(self)
            self.groups = Groups(self)

        # Lazy
        self.segs = Segments(self)

    def dump(self):

        # Dump buffers
        segs = self.segs.dump()
        groups = self.groups.dump()
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

        l = len(groups)
        self.groups.bufloc = BufferLocation(offset, len(groups))
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

    def add(self, iid, seg, group=None):
        """

        :param iid:    (obj) IID, required
        :param seg:    (obj) Segment, required
        :param group:  (str) Group name
        """

        key = len(self.lut.entries)
        iid.key = key
        seg.key = key

        entry = self.lut.add(key, iid, seg)

        if group:
            self.groups.add(name=group, entries=[entry])

    def fetch(self, everything=False, keys=None, groups=None, iids=False, segs=False):
        """Lazy loads entries from file

        :param everything:  (bool) load everything
        :param keys:        (list) keys to fetch
        :param groups:      (list) groups to fetch
        :param iids:        (bool) fetches iids
        :param segs:        (bool) fetches segments
        :return:            (list) of entries fetched
        """

        if everything:

            keys = [entry.key for entry in self.lut.entries]
            iids = True
            segs = True
            self.groups.fetch(self.groups.entries)

        else:

            if isinstance(keys, int):
                keys = [keys]

            if isinstance(groups, str):
                groups = [groups]

            keys = keys if keys is not None else []

            if groups:
                keys += list(chain.from_iterable([group.keys() for group in self.groups.fetch(groups)]))

        keys = set(keys)

        if segs:
            self.segs.fetch(keys)
        if iids:
            self.iids.fetch(keys)

        self.lut.fetched.update(keys)

        return [self.lut.entries[key] for key in keys]

    def find(self, iids, groups=None, is_hex=False):
        """Looks for matching iids in file

        :param iids:    (str|list) iids to look for
        :param groups:  (str|list) limit search to groups
        :param is_hex:  (bool) should iids be parsed as hex formatted byte strings?
        :returns:       (list) { key, iid }
        """

        if not isinstance(iids, list):
            iids = [iids]

        if isinstance(groups, str):
            groups = [groups]

        if groups:
            keys = list(chain.from_iterable([group.keys() for group in self.groups.fetch(groups)]))
        else:
            keys = [entry.key for entry in self.lut.entries]

        # TODO: currently ignoring domain, how should this be included?
        entries = self.fetch(keys=keys, iids=True)
        matches = [entry for entry in entries if entry.iid.iid in iids]

        return matches

    def group(self, name, entries=None, keys=None, iids=None, segs=None):
        self.groups.add(name, entries=entries, keys=keys, iids=iids, segs=segs)


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
        self.bufloc_groups = self.iidfile.groups.bufloc
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

    def __init__(self, iidfile):

        self.iidfile = iidfile
        self.entries = []
        self.fetched = set()

        if iidfile.exists:
            self.bufloc = iidfile.header.bufloc_lut
            self.mmap = iidfile.mmap
            self.load()

    def add(self, key, iid, seg):
        entry = LookupTableEntry(key, iid, seg)
        self.entries.append(entry)
        return entry

    def load(self):
        o, l = self.bufloc.offset, self.bufloc.offset + self.bufloc.length
        for o in range(o, l, LookupTableEntry.length):
            key = len(self.entries)
            self.entries.append(LookupTableEntry(key=key, buffer=self.mmap[o:o+LookupTableEntry.length]))

    def dump(self):
        buffer = b''.join([entry.dump() for entry in self.entries])
        return buffer


class LookupTableEntry:

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

    def __init__(self, iid=None, domain=None, key=None, bufloc=None):

        self.iid = iid
        self.domain = domain
        self.key = key
        self.bufloc = bufloc

    def load(self, buf):

        self.key, dom_len, iid_len = unpack("III", buf[:uint32*3])
        o = uint32*3
        self.domain = None if dom_len == 0 else buf[o:o+dom_len]
        o = o + dom_len
        self.iid = buf[o:o+iid_len]

    def dump(self, offset=None):

        iid = dump_str(self.iid)
        dom = dump_str(self.domain)
        buf = pack("I", self.key) + pack("I", len(dom)) + pack("I", len(iid)) + dom + iid

        if offset is not None:
            self.bufloc = BufferLocation(offset, len(buf))

        return buf


class Groups:

    def __init__(self, iidfile):

        self.iidfile = iidfile
        self.entries = {}

        if iidfile.exists:
            self.bufloc = iidfile.header.bufloc_groups
            self.mmap = iidfile.mmap
            self.load()

    def add(self, name, entries=None, keys=None, iids=None, segs=None):

        try:
            group = self.entries[name]
        except KeyError:
            group = Group(name)

        if keys:
            group.add([self.iidfile.lut.entries[k] for k in keys])

        if entries:
            group.add(entries)

        if iids:
            group.add([self.iidfile.lut.entries[iid.key] for iid in iids])

        if segs:
            group.add([self.iidfile.lut.entries[seg.key] for seg in segs])

        self.entries[name] = group

        return group

    def fetch(self, groups):

        for name in groups:
            group = self.entries[name]
            o, l = group.bufloc.offset, group.bufloc.length
            group.load(buf=self.mmap[o:o+l], lut=self.iidfile.lut)

        return [self.entries[name] for name in groups]

    def load(self):

        o, l = self.bufloc.offset, self.bufloc.length
        buf = self.mmap[o:o+l]

        length, = unpack("I", buf[:uint32])
        groups = json.loads(buf[uint32:uint32+length].decode('utf-8'))

        o += uint32+length
        for group in groups:
            name, offset, length = group['name'], group['offset'] + o, group['length']
            self.entries[name] = Group(name, bufloc=BufferLocation(offset, length))

    def dump(self):

        offset = 0
        buffer, groups = [], []
        for _, group in self.entries.items():
            buf = group.dump()
            buffer.append(buf)
            groups.append({'name': group.name, 'offset': offset, 'length': len(buf)})
            offset += len(buf)

        buffer = b''.join(buffer)
        groups = json.dumps(groups).encode('utf-8')

        return pack("I", len(groups)) + groups + buffer

    def list(self):
        """List group names

        :return:  (list) group names
        """

        return sorted(self.entries.keys())

    def get(self, groups):
        """Get entries in groups

        :param groups:  (list) group names
        :return:        (list) lut entries
        """

        if isinstance(groups, str):
            groups = [groups]

        out = []
        for group in groups:
            out.extend(list(self.entries[group].entries))

        return out


class Group:

    def __init__(self, name, entries=None, bufloc=None):

        self.name = name
        self.entries = set(entries) if entries else None
        self.bufloc = bufloc

    def add(self, entries):
        if self.entries is None:
            self.entries = set(entries)
        else:
            self.entries.update(entries)

    def keys(self):
        return [entry.key for entry in list(self.entries)] if self.entries else None

    def load(self, buf, lut):
        self.add([lut.entries[k] for k in unpack("%sI" % (len(buf) // uint32), buf)])

    def dump(self):
        return pack("%sI" % len(self.entries), *[entry.key for entry in list(self.entries)])


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

        [seg.load(buf=buf(self.mmap, seg.bufloc, offset)) for seg in segs]

    def dump(self):
        offset = 0
        buffer = []
        for entry in self.iidfile.lut.entries:
            buf = entry.seg.dump(offset=offset)
            buffer.append(buf)
            offset += len(buf)

        return b''.join(buffer)


class Segment:

    def __init__(self, key=None, bbox=None, area=None, regions=None, bufloc=None):

        self.key = key
        self.bbox = bbox
        self.area = area
        self.regions = regions
        self.bufloc = bufloc

    def load(self, buf):
        s = uint32+uint16*4+uint32
        self.key, a, b, c, d, self.area = unpack("IHHHHI", buf[:s])
        self.bbox = (a, b, c, d)
        self.regions = Regions()
        self.regions.load(buf[s:])

    def dump(self, offset=None):

        regs = self.regions.dump()
        buf = pack("IHHHHI", self.key, *self.bbox, self.area)
        buf += regs

        if offset is not None:
            self.bufloc = BufferLocation(offset, len(buf))

        return buf


class Regions:

    def __init__(self, regions=None):

        self.entries = [] if regions is None else regions

    def load(self, buf):
        o = 0
        while o < len(buf):
            length, = unpack("I", buf[o:o+uint32])

            reg = Region()
            reg.load(buf[o:o+length])
            self.entries.append(reg)

            o += length

    def dump(self):
        return b''.join([reg.dump() for reg in self.entries])


class Region:

    def __init__(self, mask=None, bbox=None):

        self.mask = mask
        self.bbox = bbox

    def load(self, buf):

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

    def dump(self):

        x = np.ndarray.flatten(self.mask)
        x = np.packbits(x)

        buffer = b'' + pack("4H", *self.bbox) + pack("%sB" % len(x), *x)
        return b'' + pack("I", len(buffer) + 4) + buffer
