#!/usr/bin/env python
# license removed for brevity
import numpy as np
import math

# Prime vertical radius of curvature
def Nrad(a,b,lat):
    e2=(pow(a,2)-pow(b,2))/pow(a,2)
    Nrad=a/pow(pow(1-e2*math.sin(lat),2),(1/2))
    return Nrad

def Marc(a,b,lat):
    f=(a-b)/a
    b0=a*(1-1/2*f+1/16*pow(f,2)+1/32*pow(f,3))

    B=b0*(lat-(3/4*f+3/8*pow(f,2)+15/128*pow(f,3))*math.sin(2*lat)+(15/64*pow(f,2)+15/64*pow(f,3))*math.sin(4*lat)-35/384*pow(f,3)*math.sin(6*lat))
    return B

def geod2TMgrid(a, b, lat, lon, lat0, lon0, scale, fnorth, feast):
    B=Marc(a,b,lat)-Marc(a,b,lat0)
    N=Nrad(a,b,lat)
    e2=(pow(a,2)-pow(b,2))/pow(a,2)

    eps2=e2/(1-e2)*pow(math.cos(lat),2)
    l=lon-lon0

    x=B+1/2*pow(l,2)*N*math.sin(lat)*math.cos(lat)+1/24*pow(l,4)*N*math.sin(lat)*pow(math.cos(lat),3)*(5-pow(math.tan(lat),2)+9*eps2+4*pow(eps2,2))
    y=l*N*math.cos(lat)+1/6*pow(l,3)*N*pow(math.cos(lat),3)*(1-pow(math.tan(lat),2)+eps2)+1/120*pow(l,5)*N*pow(math.cos(lat),5)*(5-18*pow(math.tan(lat),2)+pow(math.tan(lat),4))

    north=x*scale
    east=y*scale

    north=north+fnorth
    east=east+feast

    return north, east

def geo2UTM(lat, long):
    # Ellipsoid (GRS80) % Ellipsoid estimated specific to GPS Lat and Long
    a = 6378137
    b = 6356752.3141

    # UTM projection (Sone 32)
    lat0 = math.radians(0)
    lon0 = math.radians(9)
    scale = 0.9996
    fnorth = 0
    feast = 500000

    # Convert from geodetic coordinates to UTM coordinates
    fix_deg = ([math.radians(lat),math.radians(long)]) # Lat, Long
    north, east = geod2TMgrid(a,b,fix_deg[0],fix_deg[1],lat0,lon0,scale,fnorth,feast)
    gps_fix_utm = ([north,east])
    return gps_fix_utm
