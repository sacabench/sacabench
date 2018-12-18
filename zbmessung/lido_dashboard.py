#!/usr/bin/env python3

pixel_map = ((0x01, 0x08),
             (0x02, 0x10),
             (0x04, 0x20),
             (0x40, 0x80))

# braille unicode characters starts at 0x2800
braille_char_offset = 0x2800

def make_2d(width, height, value):
    return [[value] * width for i in range(height)]

def make_pic(width, height):
    pic = make_2d(width, height, False)
    return pic

def get_size(pic):
    height = len(pic)
    width = len(pic[0])
    return (width, height)

def draw(pic, crop_width = None, crop_height = None):
    (width, height) = get_size(pic)
    if crop_height:
        height = crop_height
    if crop_width:
        width = crop_width

    pic_s = make_2d(int((width / 2) + 1), int((height / 4) + 1), braille_char_offset)

    for y in range(0, height):
        line = pic[y]
        for x in range(0, width):
            pixel = line[x]
            pix = pixel_map[y % 4][x % 2]
            if pixel:
                pic_s[int(y / 4)][int(x / 2)] |= pix
            else:
                pic_s[int(y / 4)][int(x / 2)] &= ~pix

    for line in pic_s:
        out = ""
        for pixel in line:
            out += "{}".format(chr(pixel))
        print(out)

def set_pixel(pic, x, y, value=True):
    (width, height) = get_size(pic)

    if x >= 0 and y >= 0 and x < width and y < height:
        pic[y][x] = value

s = 50
pic = make_pic(100, 100)
for x in range(0, 100):
    for y in range(0, 100):
        if (x*x + y*y) <= (s*s):
            set_pixel(pic, x, y, True)
        #else:
            #set_pixel(pic, x, y, False)
#draw(pic)

def parse(sq_out):
    queue_jobs = []
    lines = list(sq_out.splitlines())
    headings = list(lines[0].split(";"))
    #print(headings)
    for line in lines[1:]:
        row = list(line.split(";"))
        queue_jobs.append({})
        for (key, value) in zip(headings, row):
            queue_jobs[-1][key] = value
    #print(queue_jobs)
    return queue_jobs

#data = parse("""JOBID;USER;STATE;TIME;NODELIST(REASON)
#1889315;smmaloeb;PENDING;0:00;(launch failed requeued held)
#1934039;smmaloeb;PENDING;0:00;(launch failed requeued held)
#1934042;smmaloeb;PENDING;0:00;(launch failed requeued held)
#1934120;smmaloeb;PENDING;0:00;(launch failed requeued held)
#1934123;smmaloeb;PENDING;0:00;(launch failed requeued held)
#""")

import re
t_re = re.compile("(([0-9]+)-)?(([0-9]+):)?([0-9]+):([0-9]+)")

import subprocess
#import os
#USER = os.path.expandvars("$USER")
import sys
extra_args = sys.argv[1:]

stdout = subprocess.run(["squeue", "--format", "%i;%u;%T;%M;%R"] + extra_args, stdout=subprocess.PIPE, encoding="utf-8").stdout
data = parse(stdout)

def parse_time(s):
    r = t_re.match(s)
    days = r.group(2) or '0'
    hours = r.group(4) or '0'
    mins = r.group(5) or '0'
    secs = r.group(6) or '0'
    return (int(days), int(hours), int(mins), int(secs))

def parse_time_to_seconds(s):
    (days, hours, mins, secs) = parse_time(s)
    return days * 24 * 60 * 60 + hours * 60 * 60 + mins * 60 + secs

min_time = parse_time_to_seconds("00:00")
max_time = parse_time_to_seconds("2:00:00")
d_width = 80*2
time_scale = (1 / max_time) * d_width

dpic = make_pic(d_width, len(data))
y = 0
for e in data:
    if e["STATE"] != "RUNNING":
        continue
    t = e["TIME"]
    r = parse_time_to_seconds(t)

    r = max(min_time, r)
    r = min(max_time, r)

    for i in range(0, int(r * time_scale)):
        set_pixel(dpic, i, y)
    y += 1
    #print(r)
draw(dpic, crop_height=y)
print("{} jobs drawn".format(y-1))

