import pyorbital
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import math
import pandas as pd
import bisect
import uuid
from enum import Enum

from fame_geometry import *
import copy
from shapely.geometry import Polygon
from shapely.plotting import patch_from_polygon

from matplotlib import colormaps as cmap

MIN_HORIZON_ANGLE_FOR_PASS_DEG = 15

class ObservationStatus(Enum):
    UNKNOWN = 0
    ALL_OBSERVATION_OPPORTUNITIES_ARE_CONFLICTING=1
    NO_OBSERVATION_OPPORTUNITIES=2
    COULD_NOT_FIND_BEST_SATELLITE=3
    SCHEDULED=4
    SUBMITTED=5
    DATA_RECEIVED=6
    TIMEOUT = 7
    CONSTELLATION_REJECTED = 8  # Constellation manager rejected broker request
    CANCELLED = 9               # Broker cancelled pass after a better/sufficient pass succeeded

requests_data_frame_columns = ['request', 'satellite', 'observation', 'uplink', 'downlink', 'status', 'data_product', 'scheduled_callback', 'unscheduled_callback', 'ready_callback']

class Event():
    """ An Event has a time and a function that is called at that time.
    """
    def __init__(self, time: dt.datetime, action_callable, name: str="", id: str=None):
        self.time = time
        self.action_callable = action_callable
        self.name = name
        if id is None:
            id = uuid.uuid4()
        self.id = id
    def __str__(self):
        return "Event {} at {}".format(self.name, self.time)
    def __repr__(self):
        return self.__str__()

class ObservationEvent(Event):
    def __init__(self, time: dt.datetime, action_callable, name: str="", id: str=None, satellite: Satellite=None, opportunity: ObservationOpportunity=None):
        """_summary_

        Args:
            time (dt.datetime): The time of the event
            action_callable (_type_): The function that is called at the time of the event.
            name (str, optional): the name of the event. Defaults to "".
            id (str, optional): a unique event ID. Defaults to None (in which case a UUID4 is generated).
            satellite (Satellite, optional): The satellite where the event occurs. Defaults to None.
            opportunity (ObservationOpportunity, optional): The observation opportunity performed by the satellite. Defaults to None.
        """
        super().__init__(time, action_callable, name, id)
        self.satellite = satellite
        self.opportunity = opportunity

class CommunicationEvent(Event):
    def __init__(self, time: dt.datetime, action_callable, name: str="", id: str=None, satellite: Satellite=None, station: Location=None, comm_pass: ObservationPass=None):
        super().__init__(time, action_callable, name, id)
        self.satellite = satellite
        self.station = station
        self.comm_pass = comm_pass

class Phenomenon(Pose):
    def __init__(self, lon_deg: float, lat_deg: float, alt_km: float, start_time: dt.datetime, end_time: dt.datetime, heading_deg: float=None, speed_kph: float=None, name: str=""):
        super().__init__(lon_deg=lon_deg, lat_deg=lat_deg, alt_km=alt_km, heading_deg=heading_deg, speed_kph=speed_kph, name=name)
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return "{}: Lon {}°, lat {}°, alt {} km, hdg {}°, speed {} km/h, start {}, end {}".format(self.name, self.lon_deg,self.lat_deg,self.alt_km, self.heading_deg, self.speed_kph, self.start_time, self.end_time)
    def __repr__(self):
        return self.__str__()

   
class World():
    def __init__(self, satellites: list=[], constellations: list = [], brokers: list = [], phenomena: list = [], events: list = []):
        self.satellites = satellites
        self.constellations = constellations
        self.brokers = brokers
        self.phenomena = phenomena
        self.events = events
        self.time = dt.datetime.min
        self.history = []
    
    def tick(self, print_forbidden_prefixes=[]):
        if len(self.events):
            _event = self.events.pop(0)

            _print_event_name = True
            for forbidden_names in print_forbidden_prefixes:
                if _event.name.startswith(forbidden_names):
                    _print_event_name = False
                    break
            if _print_event_name:
                print("Executing {}".format(_event))

            self.time = _event.time
            outcome = _event.action_callable()

            self.history.append({
                'time': _event.time,
                'event': _event,
                'phenomena': [copy.deepcopy(p) for p in self.phenomena],
                'states': {
                    'satellites': [copy.deepcopy(s) for s in self.satellites],
                    # Constellations and brokers have a pointer to World, which has a pointer to constellations, which...recursion! This should be fixed with new, custom deepcopy classes
                    'constellations': [copy.deepcopy(c) for c in self.constellations],
                    'brokers': [copy.deepcopy(b) for b in self.brokers],
                }
            })
            
        else:
            print("No more events")
            
        return len(self.events)

    def add_satellite(self, satellite):
        self.satellite.append(satellite)
    def add_phenomenon(self,phenomenon):
        self.phenomena.append(phenomenon)

    def add_constellation(self, constellation):
        self.constellations.append(constellation)
    
    def add_broker(self, broker):
        self.brokers.append(broker)
    
    def add_event(self, event):
        # print("Adding {}".format(event))
        if (event.time<self.time):
            raise ValueError("Event {} is earlier than sim time {}".format(event, self.time)) 
        bisect.insort(self.events, event, key=lambda x: x.time)
        
    def do_observation(self, observation: ObservationOpportunity, phenomenon_processor=lambda o, s, p: p):
        # Find phenomena close to the observation location in space and at the right time
        # Return a data product and a list of event states
        # print("Obs opp {}".format(observation))



        observed_phenomena = []

        spacecraft = observation.satellite

        if observation.instrument not in spacecraft.instruments:
            raise ValueError(f"Can't perform observation {observation} on spacecraft {spacecraft} with instruments {spacecraft.instruments} (FOVs: {spacecraft.instrument_fov_rad})")


        # Now let's see what we observed
        for _phenomenon in self.phenomena:
            if (_phenomenon.start_time <= observation.time and _phenomenon.end_time > observation.time):
                # # Compute phenomenon location on the planet
                # _ph_location_ecf = np.array(pyorbital.astronomy.observer_position(observation.time, _phenomenon.lon_deg, _phenomenon.lat_deg, _phenomenon.alt_km)[0][:3])
                # # Compute observation location on the planet
                # _obs_location_ecf = np.array(pyorbital.astronomy.observer_position(observation.time, observation.lon_deg, observation.lat_deg, observation.alt_km)[0][:3])
                # # Compute SC location
                # _sc_location_ecf = np.array(spacecraft.orbit.get_position(observation.time, normalize=False)[0][:3])
                # # Compute angle between sc-observation and sc-phenomenon
                # _ph_sc_vector = _ph_location_ecf-_sc_location_ecf
                # _obs_sc_vector = _obs_location_ecf - _sc_location_ecf
                # # print("PhSc {} || ObsSc {}".format(_ph_sc_vector, _obs_sc_vector))
                # # print("Dot: {} || N1: {} N2: {}".format(np.dot(_ph_sc_vector,_obs_sc_vector),np.linalg.norm(_ph_sc_vector,2), np.linalg.norm(_obs_sc_vector,2))) 
                # # print("Acos: {}, angle: {}".format(np.dot(_ph_sc_vector,_obs_sc_vector)/(np.linalg.norm(_ph_sc_vector,2)*np.linalg.norm(_obs_sc_vector,2)), np.arccos(np.dot(_ph_sc_vector,_obs_sc_vector)/(np.linalg.norm(_ph_sc_vector,2)*np.linalg.norm(_obs_sc_vector,2)))))
                # _ph_obs_angle_rad = np.arccos(np.clip(np.dot(_ph_sc_vector,_obs_sc_vector)/(np.linalg.norm(_ph_sc_vector,2)*np.linalg.norm(_obs_sc_vector,2)),-1,1))
                # # print("Obs angle (rad) {}".format(_ph_obs_angle_rad))
                # # If angle<FOV, return phobservation
                # if _ph_obs_angle_rad < spacecraft.instrument_fov_rad[observation.instrument]:
                if is_lla_in_satellite_fov(observation=observation, location=_phenomenon): # Note that Phenomenon inherits from Pose, which inherits from Location
                #     # print("Close enough")
                    observed_phenomena.append(_phenomenon)
                else:
                    # print("Too far")
                    pass
        # Store SOMETHING for the completed observation
        if observation not in spacecraft.data_products.keys():
            spacecraft.data_products[observation] = []
        spacecraft.data_products[observation].extend([phenomenon_processor(observation, spacecraft, p) for p in observed_phenomena])
        spacecraft.known_phenomena.append(observed_phenomena)
        
        return True
    


def do_downlink(spacecraft: Satellite, scheduler, comm_pass: ObservationPass): # scheduler is a ConstellationGroundScheduler,defined next 
    # Simple: downlink all. Future: downlink up to x.
    
    # Duration is unused for now
    # duration = comm_pass.fall.time - comm_pass.rise.time

    _downlinked = []
    if len(spacecraft.data_products):
        print("Spacecraft {} has {} data products to download".format(spacecraft, len(spacecraft.data_products)))
        
    for _observation, data_product in spacecraft.data_products.items():
        print("  Downlinked {}".format(_observation))
        _downlinked.append(_observation)
        
        # TODO here we assume the DP is for a given observation
        matching_requests = scheduler._requests[scheduler._requests['observation']==_observation]
        # print(scheduler._requests)
        # print(matching_requests)
        if len(matching_requests)>=1:
            # print(data_product)
            # try:
            scheduler._requests.loc[scheduler._requests['observation']==_observation, 'status'] = ObservationStatus.DATA_RECEIVED
            # scheduler._requests.loc[scheduler._requests['observation']==_observation, 'data_product'] = data_product
            for _ix, _ready_callback in scheduler._requests.loc[scheduler._requests['observation']==_observation, 'ready_callback'].items():
                scheduler._requests.loc[_ix, 'data_product'] = data_product
                _ready_callback(data_product)
            # except Exception as e:
            #     import pdb; pdb.set_trace()
        else:
            print("   Could not find matching request for observation {}".format(_observation))
        if len(matching_requests)>1:
            print("   I found multiple requests for observation {}!".format(_observation))

        

    for _observation in _downlinked:
        spacecraft.data_products.pop(_observation)

## History

def retell_history(world: World):
    for _chronicle in world.history:
        print("Time: {}. Event: {}".format(_chronicle['time'], _chronicle['event']))
        if type(_chronicle['event'])==ObservationEvent:
            print("Observation: sat {} and opportunity {}".format(_chronicle['event'].satellite, _chronicle['event'].opportunity))
        if type(_chronicle['event'])==CommunicationEvent:
            print("Communication: station {} to sat {} during pass {}".format(_chronicle['event'].station, _chronicle['event'].satellite, _chronicle['event'].comm_pass))

def plot_event(
        _chronicle: dict,
        world: World,
        ax=None,
        satellite_colors: dict={},
        satellite_markers: dict={},
        constellation_colors: dict={},
        plot_time: bool=True,
        plot_phenomena: bool=True,
        plot_phenomena_alpha: float=1.,
        plot_ground_stations: bool=True,
        plot_satellites: bool=True,
        plot_satellite_tracks: bool=True,
        plot_observation_gaze: bool=True,
        plot_observation_target: bool=True,
        plot_observation_footprint: bool=True,
        plot_comm_gaze: bool=True,
        plot_comm_station: bool=True,
        phenomena_colors: dict = {},
        ):
    if ax is None:
        figglobal = plt.figure(figsize=(10,5))
        ax = figglobal.add_subplot(1,1,1, projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines()

    _time_to_plot_ground_track = dt.timedelta(seconds=3*60)
    _dt_to_plot_ground_track = dt.timedelta(seconds=5)

    time_steps_for_plotting = [_chronicle['time']- _dt_to_plot_ground_track*i for i in range(int(math.ceil(_time_to_plot_ground_track/_dt_to_plot_ground_track)))]

    if plot_time:
        ax.text(0,0,"{}".format(_chronicle['time']), transform=ax.transAxes)
    
    constellation_palette = cmap['viridis'].resampled(len(world.constellations))

    if plot_phenomena:
        for phenomenon in _chronicle['phenomena']:
            if (phenomenon.start_time<_chronicle['time'] and phenomenon.end_time>_chronicle['time']): 
                ax.plot(phenomenon.lon_deg, phenomenon.lat_deg, 'D', transform=ccrs.PlateCarree(), color=phenomena_colors.get(phenomenon.name,'m'))

    for constellation_ix, constellation in enumerate(world.constellations):
        constellation_color = constellation_colors.get(constellation.name, constellation_palette(constellation_ix/len(world.constellations)))
        

        # Plot the ground stations
        if plot_ground_stations:
            for ground_station in constellation.ground_stations:
                ax.plot(float(ground_station.lon_deg), float(ground_station.lat_deg), '*', transform=ccrs.PlateCarree(), color=constellation_color)

        # Plot the satellites
        if plot_satellite_tracks:
            for satellite in constellation.satellites:
                satellite_color = satellite_colors.get(satellite.name, constellation_color)
                _orbit = satellite.orbit
                _llas = [_orbit.get_lonlatalt(t) for t in time_steps_for_plotting]
                # ax.plot([lla[0] for lla in _llas],[lla[1] for lla in _llas],transform=ccrs.Geodetic(), color=constellation_color)
                for lla_ix in range(len(time_steps_for_plotting)-1):
                    ax.plot([_llas[lla_ix+1][0], _llas[lla_ix][0]], [_llas[lla_ix+1][1], _llas[lla_ix][1]], transform=ccrs.Geodetic(), color=satellite_color, alpha = 1-(lla_ix+1)/len(_llas))

    if type(_chronicle['event'])==ObservationEvent:
        
        # Where are we looking - dashed line
        if plot_observation_gaze:
            ax.plot(
                [_chronicle['event'].opportunity.lon_deg, _chronicle['event'].satellite.orbit.get_lonlatalt(_chronicle['time'])[0]],
                [_chronicle['event'].opportunity.lat_deg, _chronicle['event'].satellite.orbit.get_lonlatalt(_chronicle['time'])[1]],
                ':k',
                transform=ccrs.Geodetic()
            )

        # The satellite location
        satellite_color  = satellite_colors.get(_chronicle['event'].satellite.name, 'b')
        satellite_marker = satellite_markers.get(_chronicle['event'].satellite.name, '.')
        
        if plot_satellites:
            ax.plot(
                _chronicle['event'].satellite.orbit.get_lonlatalt(_chronicle['time'])[0],
                _chronicle['event'].satellite.orbit.get_lonlatalt(_chronicle['time'])[1],
                satellite_marker,
                color=satellite_color,
                markersize=10,
                transform=ccrs.Geodetic()
            )

        # The center of the observation area
        if plot_observation_target:
            ax.plot(
                _chronicle['event'].opportunity.lon_deg,
                _chronicle['event'].opportunity.lat_deg,
                'Dr',
                # markersize=10,
                transform=ccrs.Geodetic()
            )

        # The sensor footprint
        if plot_observation_footprint:
            ground_footprint_llas = spacecraft_fov(
                _chronicle['time'],
                _chronicle['event'].satellite,
                _chronicle['event'].opportunity.instrument,
                _chronicle['event'].opportunity
                )
            ground_footprint_poly = Polygon([(_lla[0], _lla[1]) for _lla in ground_footprint_llas])
            ax.add_patch(patch_from_polygon(ground_footprint_poly, fc=satellite_color, ec='none', alpha=0.5, transform=ccrs.Geodetic()))

    elif type(_chronicle['event'])==CommunicationEvent:
        # The line from the GS to the satellite
        if _chronicle['event'].station != "ISL":
            if plot_comm_gaze:
                ax.plot(
                    [_chronicle['event'].station.lon_deg, _chronicle['event'].satellite.orbit.get_lonlatalt(_chronicle['time'])[0]],
                    [_chronicle['event'].station.lat_deg, _chronicle['event'].satellite.orbit.get_lonlatalt(_chronicle['time'])[1]],
                    '-.k',
                    transform=ccrs.Geodetic()
                )
            # The station
            if plot_comm_station:
                ax.plot(
                    _chronicle['event'].station.lon_deg,
                    _chronicle['event'].station.lat_deg,
                    '*',
                    markersize=10,
                    transform=ccrs.Geodetic()
                )
        # And the satellite
        if plot_satellites:
            satellite_color  = satellite_colors.get(_chronicle['event'].satellite.name, 'b')
            satellite_marker = satellite_markers.get(_chronicle['event'].satellite.name, '.')
            ax.plot(
                _chronicle['event'].satellite.orbit.get_lonlatalt(_chronicle['time'])[0],
                _chronicle['event'].satellite.orbit.get_lonlatalt(_chronicle['time'])[1],
                satellite_marker,
                color=satellite_color,
                markersize=10,
                transform=ccrs.Geodetic()
            )
    return ax




def plot_history(
        world: World,
        events_to_show: list,
        axes_extents: tuple=None,
        satellite_colors: dict={},
        satellite_markers: dict={},
        constellation_colors: dict={},
        plot_time: bool=True,
        plot_phenomena: bool=True,
        plot_phenomena_alpha: float=1.,
        plot_ground_stations: bool=True,
        plot_satellites: bool=True,
        plot_satellite_tracks: bool=True,
        plot_observation_gaze: bool=True,
        plot_observation_target: bool=True,
        plot_observation_footprint: bool=True,
        plot_comm_gaze: bool=True,
        plot_comm_station: bool=True,
        save_figure: bool=True,
        save_prefix: str="History_",
        use_sequential_index: bool=True,
        position_filter_parameters: dict={},
        ):
    artists = []
    plot_ix = -1
    for _chronicle_ix, _chronicle in enumerate(world.history):
        if type(_chronicle['event']) in events_to_show:
            figglobal = plt.figure(figsize=(20,10))
            ax = figglobal.add_subplot(1,1,1, projection=ccrs.Robinson())
            if axes_extents is None:
                ax.set_global()
            
            else:
                ax.set_extent(axes_extents, crs=ccrs.PlateCarree())
            
            ax.coastlines()

            _ax = plot_event(
                _chronicle,
                world,
                ax=ax,
                satellite_colors=satellite_colors,
                satellite_markers=satellite_markers,
                constellation_colors=constellation_colors,
                plot_time=plot_time,
                plot_phenomena=plot_phenomena,
                plot_phenomena_alpha=plot_phenomena_alpha,
                plot_ground_stations=plot_ground_stations,
                plot_satellites=plot_satellites,
                plot_satellite_tracks=plot_satellite_tracks,
                plot_observation_gaze=plot_observation_gaze,
                plot_observation_target=plot_observation_target,
                plot_observation_footprint=plot_observation_footprint,
                plot_comm_gaze=plot_comm_gaze,
                plot_comm_station=plot_comm_station,
                )
            
            # try:
            #     for broker_ix, broker in enumerate(_chronicle['states']['brokers']):
            #         if broker.name in position_filter_parameters.keys():
            #             # print(f"Plotting distribution for broker {broker.name}")
            #             # A dict with (time, keys) pairs as generated by the broker
            #             vehicle_hex_distribution = propagate_distribution_from_observations(
            #                 query_times=[_chronicle['time'],],
            #                 initial_pose=position_filter_parameters[broker.name]['initial_toi_pose'],
            #                 observation_requests=broker.workflow.constrained_observation_requests,
            #                 pose_propagator=lambda _previous_pose, _deltat: ship_propagator(
            #                     previous_pose=_previous_pose,
            #                     deltat=_deltat,
            #                     heading_variance=position_filter_parameters[broker.name]['heading_variance'],
            #                     speed_variance=position_filter_parameters[broker.name]['speed_variance'],
            #                     max_speed_if_unspecified_kph=position_filter_parameters[broker.name]['max_speed_if_unspecified_kph'],
            #                     ),
            #                 target_name=position_filter_parameters[broker.name]['target_name'],
            #                 h3_resolution=5, # Res 5: 9.85 km side. Res 6: 3.72 km side
            #                 # samples_propagation: int=200,
            #                 verbose=False,
            #                 plot_filter_distribution=False,
            #                 propagate_negative_samples=True,
            #             )
            #             # This will produce a distribution for _unfulfilled_ requests in the future. 
            #             # It tells us where the broker will be looking next.
            #             # Find the closest time step to the current time
            #             if len(vehicle_hex_distribution.keys()):
            #                 closest_future_timestep = min(vehicle_hex_distribution.keys(), key=lambda t: t-_chronicle['time'] if t>_chronicle['time'] else dt.timedelta.max)
            #                 if closest_future_timestep<_chronicle['time']:
            #                     raise ValueError("The filter is looking in the past, that is very odd")
            #             # Plot that
            #             vehicle_hex_distribution[closest_future_timestep].plot(
            #                 ax=ax,
            #                 transform=ccrs.PlateCarree(), 
            #                 column='point_count',
            #                 cmap=position_filter_parameters[broker.name].get('color_scheme', 'Blues'),
            #                 alpha=.5,
            #                 # vmax=points_hex_tessellation['point_count'].max() + 1
            #                 )
            #         # else:
            #         #     print(f"Broker {broker.name} not in plotting keys")
            # except ValueError as e:
            #     continue


            if use_sequential_index:
                plot_ix+=1
            else:
                plot_ix = _chronicle_ix
            if save_figure:
                plt.savefig("{}{:05d}.png".format(save_prefix, plot_ix), bbox_inches='tight')
            # artists.append(_ax)
                plt.close()
        

## Utilities

def screen_opportunity_for_feasibility(existing_requests: pd.DataFrame, satellite: Satellite, _request: ObservationOpportunity, screen_against_comm_passes:bool=True, log_prefix: str=""):
    # Check if a given request conflicts with existing requests.
    # TODO this is horrifyingly expensive because we do not exploit the fact that
    #  requests are sorted. We should improve this, ideally without rebuilding a full on timeline library.
    # Idea: have a timeline per satellite that says "is this sat occupied". 
    # The underlying object is a dict with satellites as keys and a list of (time, +-1) as values (ie a timeline). +1 occupied, -1 free.
    # When checking if a request is free, we want to check that:
    # - The start and end of a task, when inserted, fall in the same slot (which means we are not splitting an existing task)
    # - The value of the timeline immediately before that slot is "free".
    # 
    conflicting_requests = existing_requests.loc[
        existing_requests.apply(
        lambda x: 
            (x['status']== ObservationStatus.SCHEDULED) and # We have actually scheduled this
            (x['observation'].time+x['observation'].duration > _request.time) and # The end of the other observation is after we start
            (x['observation'].time < _request.time+_request.duration) and # The start of the other observation is before we end
            (x['satellite'] == satellite) # This request is on the same satellite. Note that we check these are the same OBJECT, not just the same name.
        , axis=1)]
    
    if len(conflicting_requests):
        print("   [{}:{}]Conflict with another request".format(log_prefix, satellite.name))
        return False
    
    if (screen_against_comm_passes):
        conflicting_uplinks = existing_requests.loc[
            existing_requests.apply(
            lambda x: 
                (x['status']==ObservationStatus.SCHEDULED) and # We have actually scheduled this
                (type(x['uplink']) == ObservationPass) and
                (x['uplink'].fall.time > _request.time) and # The end of the comm pass is after we start
                (x['uplink'].rise.time < _request.time + _request.duration) and # The start of the comm pass is before we end
                (x['satellite'] == satellite) # This request is on the same satellite
            , axis=1)]
        if len(conflicting_uplinks):
            print("   [{}:{}]Conflict with an uplink".format(log_prefix, satellite.name))
            return False
        conflicting_downlinks = existing_requests.loc[
            existing_requests.apply(
            lambda x: 
                (x['status']==ObservationStatus.SCHEDULED) and # We have actually scheduled this
                (type(x['downlink']) == ObservationPass) and
                (x['downlink'].fall.time > _request.time) and # The end of the comm pass is after we start
                (x['downlink'].rise.time < _request.time + _request.duration) and # The start of the comm pass is before we end
                (x['satellite'] == satellite) # This request is on the same satellite
            , axis=1)]
        if len(conflicting_downlinks):
            print("   [{}:{}]Conflict with a downlink".format(log_prefix, satellite.name))
            return False
    return True

def screen_pass_for_feasibility(existing_requests: pd.DataFrame, satellite: Satellite,  _obs_pass: ObservationPass, screen_against_comm_passes:bool=False, log_prefix: str=""):
    # Check if a given pass conflicts with existing requests.
    # TODO this is horrifyingly expensive because we do not exploit the fact that
    #  requests are sorted. We should improve this, ideally without rebuilding a full on timeline library.
    conflicting_requests = existing_requests.loc[
        existing_requests.apply(
        lambda x: 
            (x['status']==ObservationStatus.SCHEDULED) and # We have actually scheduled this
            (x['observation'].time+x['observation'].duration > _obs_pass.rise.time) and # The end of the other observation is after we start
            (x['observation'].time < _obs_pass.fall.time) and # The start of the other observation is before we end
            (x['satellite'] == satellite) # This request is on the same satellite. Note that we check these are the same OBJECT, not just the same name.
        , axis=1)]
    if len(conflicting_requests):
        return False
    if (screen_against_comm_passes):
        conflicting_uplinks = existing_requests.loc[
            existing_requests.apply(
            lambda x: 
                (x['status']==ObservationStatus.SCHEDULED) and # We have actually scheduled this
                (type(x['uplink']) == ObservationPass) and
                (x['uplink'].fall.time > _obs_pass.rise.time) and # The end of the comm pass is after we start
                (x['uplink'].rise.time < _obs_pass.fall.time) and # The start of the comm pass is before we end
                (x['satellite'] == satellite) # This request is on the same satellite
            , axis=1)]
        if len(conflicting_uplinks):
            return False
        conflicting_downlinks = existing_requests.loc[
            existing_requests.apply(
            lambda x: 
                (x['status']==ObservationStatus.SCHEDULED) and # We have actually scheduled this
                (type(x['downlink']) == ObservationPass) and
                (x['downlink'].fall.time > _obs_pass.rise.time) and # The end of the comm pass is after we start
                (x['downlink'].rise.time < _obs_pass.fall.time) and # The start of the comm pass is before we end
                (x['satellite'] == satellite) # This request is on the same satellite
            , axis=1)]
        if len(conflicting_downlinks):
            return False
    return True
