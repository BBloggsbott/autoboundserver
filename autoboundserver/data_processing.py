import time
import datetime
import math

def get_time_stamp():
    time_zone = time.timezone/3600
    frac, whole = math.modf(time_zone)
    print(frac, whole)
    if whole<0:
        time_zone = "+{}:{}".format(int(-whole), int(-frac*60))
    else:
        time_zone = "{}:{}".format(int(-whole), int(frac*60))
    timestamp = '{:%Y-%m-%dT%H:%M:%S}{}'.format(datetime.datetime.now(), time_zone)
    return timestamp

def generate_osm_xml(points, id):
    timestamp=get_time_stamp()
    head = '''<?xml version="1.0" encoding="UTF-8"?>
            <osm version="0.6" generator="JOSM">
                <way id="{0}" visible="true" >'''.format(id)
    tail='<tag k="building" v="yes" /></way></osm>'
    ways_xml = ""
    id-=1
    for i in points:
        ways_xml+='''<node id="{0}" east="{1} north="{2}" />'''.format(id, i[0], i[1])
        id-=1
    osm_xml = head+ways_xml+tail
    return osm_xml, id

def invert_approx(approx):
    inverted_approx = [i[0][::-1] for i in approx]
    return invert_approx

def unsqueeze_approx(approx):
    unsqueezed_approx = [i[0] for i in approx]
    return unsqueezed_approx

def offset_approx(unsqueezed_approx, x, y):
    offset_aprx = [(i[0]+x, i[1]+y) for i in unsqueezed_approx]
    return offset_aprx