###############################################################################
# Python script used to read in LeCroy spreadsheet view and create
# a CSV file showing count/timing metrics on child tags in the USB capture.
#
# NOTE: The LeCroy file is an export CSV file from the spreadsheet view
# with default columns.  All transfers and data must be expanded with
# multiple transactions per transfer.

###############################################################################
# region Import region
import os
import sys
import numpy as np
import pandas as pd
import argparse  # command line parser
import collections
from datetime import timedelta
import matplotlib.pyplot as plt
import logging
import tags
import yaml
import timeit
import inspect
# endregion

###############################################################################
# region Variables region
###############################################################################
__version__ = '1.1'  # version of script

# global di ctionary containing all child tags encountered
# in this capture along with number of occurences.
g_children_found = collections.Counter()
g_child_payload_tally = collections.Counter()

# Required columns in the LeCroy CSV file.
# Note this file is a LeCroy export of the spreadsheet
# view with all Transfers and data expanded.
REQUIRED_COLUMNS = ['Time Stamp', 'Item', 'Addr,Endp,Dir', 'Data']

# Minimum number of rows required
MIN_ROW_COUNT = 10

# I know what I'm doing
pd.options.mode.chained_assignment = None  # default='warn'

# Column for our metrics CSV file.
g_metric_columns = ['Parent Tag', 'Child Tag', 'Endp', 'Dir', 'Child Tag Count', 'Child Length Total', 'Ratio',
                    'Mean (mS)', 'Median (mS)', 'Max (mS)', 'Min (mS)', 'Std', 'Mean Child SPS', 'Max Child SPS',
                    'Min Child SPS',
                    'Sample Drift Measured MAX Sec',
                    'Sample Drift Measured MAX ppm', 'Sample Drift Measured MIN Sec', 'Sample Drift Measured MIN ppm',
                    'Sample Drift Measured MEAN', 'Sample Drift Measured STD', 'Sample Drift Expected seconds',
                    'Sample Drift Samples Collected']


# These particular columns will be changed to an unsigned integer data type
g_metrics_df_int_value_list = [
    'Endp',
    'Child Tag Count',
    'Child Length Total',
    'Sample Drift Expected seconds',
    'Sample Drift Samples Collected'
]

# These particular columns will be changed to float
g_metrics_df_float_value_list = [
    'Mean (mS)',
    'Median (mS)',
    'Max (mS)',
    'Min (mS)',
    'Std',
    'Mean Child SPS',
    'Max Child SPS',
    'Min Child SPS',
    'Sample Drift Measured MAX Sec',
    'Sample Drift Measured MAX ppm',
    'Sample Drift Measured MIN Sec',
    'Sample Drift Measured MIN ppm',
    'Sample Drift Measured MEAN',
    'Sample Drift Measured STD',
]

# Filler for cells that were not calculated
G_NOT_APPLICABLE = 0
G_NOT_ENOUGH_SAMPLES = 0


# Marker flag defines
QRS_COMPLEX_DETECTED_MASK = 0x8000
QRS_SAMPLE_INDEX_MASK = 0x0700
QRS_SAMPLE_INDEX_SHIFT = 8
QRS_MARKER_OFFSET = 500


START_TLD_SEQUENCE = 70000

# endregion

###############################################################################
# region Function region
###############################################################################


def to_device(tld_hex_string):
    """Return the device name of the given TLD.

    Args:
        tld_hex_string (): The TLD in hex string format.

    Returns:
        The device name string based on the hex value from the TLD hex string.
        Note the device name string is from /etram/common_xml.

    """
    try:
        return tags.DeviceId_Dict[int(tld_hex_string[4:8], 16)]
    except:
        logging.warning('could not find device in tld data: %s', tld_hex_string, exc_info=True)
        return None


def to_payload_length(tld_hex_string):
    """Return the payload length of the given TLD.

    Args:
        tld_hex_string (): The TLD in hex string format.

    Returns:
        The decimal value of the TLD length number parsed out of the given TLD string.

    """
    try:
        return int(tld_hex_string[8:12], 16)
    except:
        logging.warning('cant find tld length in tld data: %s', tld_hex_string, exc_info=True)
        return 0


def to_sequence(tld_hex_string):
    """Return the sequence number of the given TLD.

    Args:
        tld_hex_string (): The TLD in hex string format.

    Returns:
        The decimal value of the TLD sequence number parsed out of the given TLD string.

    """
    try:
        return int(tld_hex_string[12:16], 16)
    except:
        logging.warning('cant find tld sequence in tld data: %s', tld_hex_string, exc_info=True)
        return 0


def to_trans(lecroy_item):
    """Return the transaction number of the given transfer string.

    Args:
        lecroy_item (): The string from the LeCroy transfer/transaction.  Note that item is the column header name from
        the lecroy CSV export.

    Returns:
        The decimal value of the transfer number.  This allows the viewer to match the transfer from the LeCroy
        output.

    """
    transfer_number = 0
    if lecroy_item.startswith('Xfr'):
        transfer_number = int(lecroy_item.split(' ')[1])  # parse out the number and make decimal
    return transfer_number


def tld_to_string(row):
    """Return the given pandas DataFrame row as a decoded string.

    The given TLD row is returned as a full string decoded down
    to parent/child tag strings.
    This string is used in subsequent metrics processing.

    We also tally 2 additional metrics:
        The number of times a particular child tag occurred.
        The total payload side for a particular child tag.

    Args:
        row (): A DataFrame row

    Returns:
        A TLD in string format.  Note that strings are from /etram/common_xml.

    """

    tld_string = ''
    child_dictionary = None
    p_string = None

    for tags_list in tld_tags(row.Data):
        # device = tags_list[0]    # index zero is the device
        parent = tags_list[1]  # index one is the parent

        # Attempt to get the parent tag...
        try:
            # d_string = device_dictionary[device]
            p_string = tags.ParentTag_Dict[parent]
        except KeyError:
            # Parent tag is not in the dictionary, so construct one to use.
            # logging.warning('Parent tag : ' + str(parent) + ' not found in list', exc_info=True)
            p_string = 'UNKNOWN_PARENT_TAG_' + str(parent)
        except Exception as e:
            logging.warning(e)
            print(e)
            exit(-1)
        finally:
            # Attempt to get the child dictionary for this parent tag...
            try:
                child_dictionary = tags.ChildTag_Dict[parent]
            except KeyError:
                # Look like there's no child dictionary for this parent.
                # Provide an UNKNOWN one.
                child_dictionary = tags.ChildTag_Dict[0]
            except Exception as e:
                logging.warning(e)
                print(e)
                exit(-1)
            finally:
                # Process all the child tags in the list
                for child_tag in tags_list[2:]:
                    # Attempt to get the tag strings...
                    try:
                        c_string = p_string + ':' + str(row.Endp) + ':' + row.Dir + ':' + child_dictionary[child_tag[0]]
                    except KeyError:
                        # Child tag is not in the dictionary, so construct one to use.
                        # logging.warning('Child tag : ' + str(child_tag[0]) + ' not found in list', exc_info=True)
                        c_string = p_string + ':' + str(row.Endp) + ':' + row.Dir + ':' + 'UNKNOWN_CHILD_TAG_' + str(
                            child_tag[0])
                    except Exception as e:  # game over man
                        logging.warning(e)
                        print(e)
                        exit(-2)
                    finally:
                        tld_string += c_string + ','  # build string...
                        g_children_found[c_string] += 1  # Keep tally of child tag hits...
                        g_child_payload_tally[c_string] += child_tag[1]

    return tld_string


def tld_tags(tld):
    """Return the given TLD as a sequence of decoded intgers in list format.

    Args:
        tld (): A hex string containing a full TLD to parse.

    Returns:
        An integer list of tags parsed from the passed in TLD.

    """
    tld_offset = 0
    tld_end = len(tld)  # end of this string
    ###########################
    while tld_offset < tld_end:  # discover all tlds
        parent_offset = 0
        if to_device(tld[tld_offset:]) is not None:  # sanity check this tld
            device_id = int(tld[tld_offset + 4:tld_offset + 8], 16)  # get the device id for this tld
            parent_offset = tld_offset + 20  # offset to first parent tag
            parent_end = parent_offset + to_payload_length(tld[tld_offset:]) * 4  # this should offset to the tld crc

            # Is the calculated transfer end past the length of the TLD?
            if parent_end + 4 > tld_end:
                logging.warning('Bad TLD found, skipping : %s', tld)
                print('Malformed TLD encountered.  Parent end is beyond TLD end.')
                return

            ################################
            # discover all parent tags
            while parent_offset < parent_end:
                t = []  # init our list we will build
                t.append(device_id)  # add device ID
                t.append(int(tld[parent_offset:parent_offset + 4], 16))  # add parent tag
                child_offset = parent_offset + 8  # offset to child tag
                parent_offset += int(tld[parent_offset + 4:parent_offset + 8], 16) * 4 + 8
                ##################################
                # discover all child tags
                while child_offset < parent_offset:
                    t.append(
                        [int(tld[child_offset:child_offset + 4], 16), int(tld[child_offset + 4:child_offset + 8], 16)])
                    child_offset += int(tld[child_offset + 4:child_offset + 8], 16) * 4 + 8  # advance to next child

                # Sanity test the child_offset.  Child offset and parent offset should equal or something is amiss
                if child_offset != parent_offset:
                    logging.warning('Bad TLD found, skipping : %s', tld)
                    print('Malformed TLD encountered.  Child end is beyond Parent end.')
                    return

                yield t  # send back list of discovered tld tags

        # Advance to next possible parent tag...
        tld_offset += parent_offset + 4  # inc past tld crc


def stripspace(string_to_strip):
    """Remove all spaces from the given string.

    Args:
        string_to_strip (): The string to remove spaces from.

    Returns:
        A string with all spaces removed.

    """
    return string_to_strip.replace(' ', '')


def print_console_and_log(message):
    """Take the give line and send it to the console and the log file.

    Args:
        message (): The string to print to console and send to log.

    Returns:
        nothing

    """
    print(message)
    logging.info(message)


def to_series(df, parent_tag, child_tag, prepend_child_count=False):
    """Take the given dataframe and pattern and find the data and return a series

    Args:
        df(): The dataframe to search.
        parent_tag: the parent tag to find
        child_tag: the parent tag to find
        prepend_child_count: boolean to pre-pend child count

    Returns:
        a series containing the discovered data

    """
    parent_string = tags.ParentTag_Dict[parent_tag]
    child_string = (tags.ChildTag_Dict[parent_tag])[child_tag]

    found = df[df['tags'].str.contains(parent_string + ':1:.*' + child_string)]
    data_series = tld_data(found['Data'], parent_tag, child_tag, prepend_child_count)
    return data_series


def to_comms_data(df):
    """
    Take the given dataframe and pattern and find the data and return a series.
    NOTE: This is extremely dependent on tld order by the U-RE.  Assumption are made
    on the position of waveform data.


    Args:
        df(): The dataframe to search.
        parent_tag: the parent tag to find
        child_tag: the parent tag to find

    Returns:
        a series containing of COMMS_DATA dictionaries

    """
    # Find end point 1, direction in, device is ure.
    # This is all the URE DSP outbound data.
    found = df[(df.Endp == 1) & (df.Dir == 'IN') & (df.Device == 'ACTIVE_CABLE_ECG_DSP')]

    # Iterate through all the data and collect...
    # There will be one COMMS_DATA structure created for each TLD packet sent.
    cd = []
    for tld in found.Data:
        tld_offset = 0
        tld_end = len(tld)  # end of this string

        # There will be ones COMMS_DATA per TLD...
        COMMS_DATA = \
        {
            'tld_seq': 0,
            'buffer_sps_500': np.zeros((9, 5), dtype=np.int16),
            'buffer_sps_500_ao': np.zeros((5,), dtype=np.int16),
            'pace_markers': 0,
            'paceInfo': [np.zeros((3,), dtype=np.int16), 0],
            'ecgCount': 0
        }

        # First thing to do is to capture the number ECG samples in this TLD.
        COMMS_DATA['ecgCount'] = int(tld[32:36], 16)
        COMMS_DATA['buffer_sps_500_ao'] = np.array([int(tld[i:i + 4], 16) for i in range(36, 56, 4)], dtype=np.int16)  # AO

        # Capture the TLD sequence number.
        COMMS_DATA['tld_seq'] = to_sequence(tld)

        # Are we 4?
        if COMMS_DATA['ecgCount'] == 4:
            # Build our COMMS_DATA dictionary base on 4 samples
            COMMS_DATA['buffer_sps_500'][0] = np.array([int(tld[i:i + 4], 16) for i in range(80, 100, 4)], dtype=np.int16)    # I
            COMMS_DATA['buffer_sps_500'][1] = np.array([int(tld[i:i + 4], 16) for i in range(104, 124, 4)], dtype=np.int16)   # II
            COMMS_DATA['buffer_sps_500'][2] = np.array([int(tld[i:i + 4], 16) for i in range(128, 148, 4)], dtype=np.int16)   # III
            COMMS_DATA['buffer_sps_500'][3] = np.array([int(tld[i:i + 4], 16) for i in range(152, 172, 4)], dtype=np.int16)   # V1
            COMMS_DATA['buffer_sps_500'][4] = np.array([int(tld[i:i + 4], 16) for i in range(176, 196, 4)], dtype=np.int16)   # V2
            COMMS_DATA['buffer_sps_500'][5] = np.array([int(tld[i:i + 4], 16) for i in range(200, 220, 4)], dtype=np.int16)   # V3
            COMMS_DATA['buffer_sps_500'][6] = np.array([int(tld[i:i + 4], 16) for i in range(224, 244, 4)], dtype=np.int16)   # V4
            COMMS_DATA['buffer_sps_500'][7] = np.array([int(tld[i:i + 4], 16) for i in range(248, 268, 4)], dtype=np.int16)   # V5
            COMMS_DATA['buffer_sps_500'][8] = np.array([int(tld[i:i + 4], 16) for i in range(272, 292, 4)], dtype=np.int16)   # V6
            COMMS_DATA['pace_markers'] = int(tld[60:64], 16)
        # gotta be 5...
        else:
            # Build our COMMS_DATA dictionary base on 5 samples
            COMMS_DATA['buffer_sps_500'][0] = np.array([int(tld[i:i + 4], 16) for i in range(84, 104, 4)], dtype=np.int16)    # I
            COMMS_DATA['buffer_sps_500'][1] = np.array([int(tld[i:i + 4], 16) for i in range(112, 132, 4)], dtype=np.int16)   # II
            COMMS_DATA['buffer_sps_500'][2] = np.array([int(tld[i:i + 4], 16) for i in range(140, 160, 4)], dtype=np.int16)   # III
            COMMS_DATA['buffer_sps_500'][3] = np.array([int(tld[i:i + 4], 16) for i in range(168, 188, 4)], dtype=np.int16)   # V1
            COMMS_DATA['buffer_sps_500'][4] = np.array([int(tld[i:i + 4], 16) for i in range(196, 216, 4)], dtype=np.int16)   # V2
            COMMS_DATA['buffer_sps_500'][5] = np.array([int(tld[i:i + 4], 16) for i in range(224, 244, 4)], dtype=np.int16)   # V3
            COMMS_DATA['buffer_sps_500'][6] = np.array([int(tld[i:i + 4], 16) for i in range(252, 272, 4)], dtype=np.int16)   # V4
            COMMS_DATA['buffer_sps_500'][7] = np.array([int(tld[i:i + 4], 16) for i in range(280, 300, 4)], dtype=np.int16)   # V5
            COMMS_DATA['buffer_sps_500'][8] = np.array([int(tld[i:i + 4], 16) for i in range(308, 328, 4)], dtype=np.int16)   # V6
            COMMS_DATA['pace_markers'] = int(tld[64:68], 16)

        # Add to bottom of list...
        cd.append(COMMS_DATA)

    return pd.Series(cd)


###############################################################################
# Calculate the time differences between ALL the surface temperature requests
# and the reponses.  We return a series of deltas.
def get_temp_delta(df):
    """Return the request/response times for the surface temperature request.

    The returned series is the delta time between a host cable surface
    temperature request and the response from the cable.  That is, how much
    time elapsed between the request and reponsed.

    Args:
        df (): The fully built dataframe containing all the information needed to calculate the time time difference
         between surface temperature request/response.

    Returns:
        A pandas series containing

    """
    # Get request/response times as series from main dataframe
    request_series = (df[df.tags.str.contains('ACTIVE_CABLE_TEMPERATURE_REQUEST,')]['Time Stamp']).reset_index(
        drop=True)
    response_series = (df[df.tags.str.contains('ACTIVE_CABLE_TEMPERATURE,')]['Time Stamp']).reset_index(drop=True)
    count_series = ((df[df.tags.str.contains('ACTIVE_CABLE_TEMPERATURE,')]).
                    tags.str.count('ACTIVE_CABLE_TEMPERATURE')).reset_index(drop=True)

    # Log the counts
    logging.info('Count of temperature requests: %i', request_series.count())
    logging.info('Count of tlds with temperature responses %i', response_series.count())
    logging.info('Count of temperature responses: %i', count_series.sum())

    # Sanity check
    if request_series.count() != 0 and response_series.count() != 0:
        # Make a dataframe from count and response series
        response_df = pd.concat([count_series, response_series], axis=1)

        adjusted_response_list = []
        for row in response_df.itertuples():
            for i in range(row.tags):
                adjusted_response_list.append(row._2)

        # Convert list to series
        adjusted_response_series = pd.Series(adjusted_response_list)

        delta_series = adjusted_response_series - request_series

        return delta_series
    else:
        return None


def tld_data(tld_packet_series, target_parent_tag, target_child_tag, prepend_child_count=False):
    """Given a TLD collection in the given pandas DataFrame, find and extract
       all the child data from the given parent/child tag.

    All the discovered child data is return in a pandas Series for a given
    parent and child tag.

    Args:
        tld_packet_series (): A pandas series containing the TLD in hex string format.
        target_parent_tag (): The decimal value of the parent tag of the data to find.
        target_child_tag ():  The decimal value of the child tag of the data to find.
        prepend_child_count (): Boolean value to indicate if child count is needed in front of child data.

    Returns:
        A series containing the decimal series of the data extracted from the parent/child tag.

    """
    data_list = []  # The list of child data we build.
    # Process all tld packets in the series
    for tld in tld_packet_series:
        tld_offset = 0
        tld_end = len(tld)  # end of this string
        ###########################
        while tld_offset < tld_end:  # walk through all the tlds
            parent_offset = 0
            if to_device(tld[tld_offset:]) is not None:  # sanity check this tld
                parent_offset = tld_offset + 20  # offset to first parent tag
                parent_end = parent_offset + to_payload_length(
                    tld[tld_offset:]) * 4  # this should offset to the tld crc

                # Is the calculated transfer end past the length of the TLD?
                if parent_end + 4 > tld_end:
                    logging.warning('Bad TLD found, skipping : %s', tld)
                    print('Bad TLD found, skipping')
                    break

                ################################
                # Walk down the tld and exam all parent tags...
                while parent_offset < parent_end:
                    # Grab the parent tag
                    parent_tag = int(tld[parent_offset:parent_offset + 4], 16)  # add parent tag

                    child_offset = parent_offset + 8  # offset to child tag
                    parent_offset += int(tld[parent_offset + 4:parent_offset + 8], 16) * 4 + 8

                    ##################################
                    # Walk down the tld and exam all child tags...
                    while child_offset < parent_offset:
                        # Grab the child tag
                        child_tag = int(tld[child_offset:child_offset + 4], 16)

                        # Did we get a match?
                        if parent_tag == target_parent_tag and child_tag == target_child_tag:
                            # Loop here, grab the child data and tack to end of series...
                            child_length = int(tld[child_offset + 4:child_offset + 8], 16)
                            data_offset = child_offset + 8

                            # Should we pre-pend the child data count to child data?
                            if prepend_child_count is True:
                                data_list.append(child_length)   # add to array...

                            # Append child data to array.
                            data_list += [int(tld[i:i + 4], 16) for i in
                                          range(data_offset, data_offset + (child_length * 4), 4)]

                        # next child...
                        child_offset += int(tld[child_offset + 4:child_offset + 8],
                                            16) * 4 + 8  # advance to next child

            # Advance to next possible tld tag...
            tld_offset += parent_offset + 4  # inc past tld crc

    return pd.Series(data_list, dtype='int')


def to_tag_values(parent, child):
    """ Return tag values for the given parent and child strings (reverse dictionary).

    :param parent: string for parent tag
    :param child: string for child tag
    :return: tuple containing values for parent and child
             None for not found
    """
    try:
        parent_value = list(tags.ParentTag_Dict.keys())[list(tags.ParentTag_Dict.values()).index(parent)]
    except ValueError:
        logging.warning('could not find given parent tag in tag dictionary: %s', parent, exc_info=True)
        return None

    try:
        child_dict = tags.ChildTag_Dict[parent_value]
        child_value = list(child_dict.keys())[list(child_dict.values()).index(child)]
    except ValueError:
        logging.warning('could not find given child tag in tag dictionary: %s', child, exc_info=True)
        return None

    return parent_value, child_value


def write_colleciton(df, tag_list, data_type, filename):
    """
    Find, extract and write the data from the given dataframe to a file.

    :param df: the dataframe to search and extract data from
    :param tag_list:  the list of tuples containing column name, parent tag, child tag
    :param data_type: the data type to convert ALL the columns to
    :param filename: the filename to write to
    :return: True if dataframe built and written to given file
             False if an error occurred
    """
    df_to_build = pd.DataFrame()  # dataframe to build
    cols = []  # column list to build
    # Rip through given list, find data and add to our dataframe
    for tag in tag_list:
        name, p, c = tag
        data_series = to_series(df, p, c)  # get the data for this tag
        # Add it to our frame
        df_to_build = df_to_build.append(data_series, ignore_index=True)
        # Build our column list
        cols.append(name)

    # Transpose frame, set column names, set correct data type and write to file
    df_to_build = df_to_build.T
    df_to_build.columns = cols
    df_to_build[cols] = df_to_build[cols].astype(data_type)
    df_to_build.to_csv(filename, index=False)


def read_yaml(yaml_filename):
    """
    Read in the given yaml file and return the structure

    :param yaml_filename: the filename of the yaml file to read
    :return: A configuration dictionary the caller understands.
    """
    config = None
    try:
        config = yaml.safe_load(open(yaml_filename))
    except IOError as e:
        print("Can't read " + args.yaml_config)
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        logging.warning('Exceptions reading YAML file %s', yaml_filename, exc_info=True)
    except:
        print('Problem parsing YAML test configuration file ' + yaml_filename + ', syntax issue?')
        logging.warning('Exceptions reading YAML file %s', yaml_filename, exc_info=True)

    return config


def write_comms_data(final, filename):
    """
    Collect and write COMMS_DATA from given dataframe to given filename.
    :param final: the dataframe containing the TLDs collection from a URE.  Assumed to be URE.  If not, not data will
    be written.
    :param filename:  The filename to write the C array to.
    :return:
    """

    # Attempt to collect COMMS_DATA from this dataframe.
    pd_series = to_comms_data(final)
    # Did we get any?
    if len(pd_series) is 0:
        return False
    # and write to file.
    with open(filename, 'w') as f:
        # Write array declaration...
        f.write('COMMS_DATA g_comms_data_capture[] = \n')
        f.write('{\n')

        # Iterate through all the collect DATA_COMMS structures...
        for d in pd_series:
            f.write('// tld sequence {:d}\n'.format(d['tld_seq']))

            f.write('{\n')
            f.write('// buffer_sps_500\n')
            for x in np.nditer(d['buffer_sps_500']):
                f.write('{}, '.format(x))
            f.write('\n')

            for x in np.nditer(d['buffer_sps_500_ao']):
                f.write('{}, '.format(x))
            f.write(' // buffer_sps_500_ao\n')

            f.write('0x{:04x},             // pace_markers\n'.format(d['pace_markers']))

            for x in np.nditer(d['paceInfo'][0]):
                f.write('{}, '.format(x))
            f.write('{},         // paceInfo\n'.format(d['paceInfo'][1]))

            f.write('{:d}                   // ecgCount\n'.format(d['ecgCount']))
            f.write('},\n\n')

        # Complete declaration.
        f.write('};\n')

    return True


# endregion

###############################################################################
# region Main region
def main(arg_list=None):
    """The main function of this module.

    Perform all the processing on a LeCroy CSV exported active cable capture.
    Returns the timing metrics on the capture data.

    """
    ##############
    # Initialize some variables
    g_metrics_series = pd.Series(index=g_metric_columns)
    g_metrics_df = pd.DataFrame(columns=g_metric_columns)
    g_children_found.clear()
    g_child_payload_tally.clear()
    G_COLLECTION_FILENAME = 'collect.yaml'
    G_COLLECTION_CONFIG = None

    ###############################################################################
    # region Command line region

    ###############################################################################
    # Setup command line argument parsing...
    parser = argparse.ArgumentParser(description="Process an exported LeCroy CSV file (from spreadsheet view)")
    parser.add_argument('-i', dest='csv_input_file',
                        help='name of exported LeCroy CSV to read and process', required=True)
    parser.add_argument('-o', dest='csv_output_file',
                        help='name of CSV output file to write statistics to', required=False)
    parser.add_argument('-c', dest='store_csv_data_file', default=False, action='store_true',
                        help='flag to create optional CSV output file to write full captured data to, this data was \
                        used to build statistics.  Useful for additional analysis', required=False)
    parser.add_argument('-g', dest='graphics_extension',
                        help='3 letter extention of optional graphics file for histogram charts', required=False)
    parser.add_argument('-d', dest='graphics_dir',
                        help='optional directroy where graphics files will be written', required=False)
    parser.add_argument('-y', dest='yaml_config', default='sample_drift_config.yaml',
                        help='File for sample drift test configuration, defaults to sample_drift_config.yaml',
                        required=False)
    parser.add_argument('-p', dest='precision', default=6,
                        help='Number of decimal places to use on generated metrics file',
                        type=int, required=False)
    parser.add_argument('--log', dest='loglevel', default='INFO', required=False,
                        help='logging level for log file')
    parser.add_argument('-t', dest='start_time_seconds', default=0.0, type=float, required=False,
                        help='seconds into collection to start analysis, defaults to 0.0')
    parser.add_argument('--collect', dest='collect',
                        help='collect given data', required=False)
    parser.add_argument('--comms_data', dest='comms_data', default=False, action='store_true',
                        help='flag to create optional C array output containing an array of COMMS_DATA types',
                        required=False)
    parser.add_argument('-v', dest='verbose', default=False, action='store_true',
                        help='verbose output flag', required=False)
    parser.add_argument('-a', dest='discard_analog', default=False, action='store_true',
                        help='AO flag.  If set, will remove AO endpoint from CSV database file', required=False)
    parser.add_argument('-s', dest='separate_streams', default=False, action='store_true',
                        help='write separate steams to CSV files', required=False)
    parser.add_argument('--version', action='version', help='Print version.',
                        version='%(prog)s Version {version}'.format(version=__version__))

    # Parse the command line arguments
    args = parser.parse_args(arg_list)

    ###############################################################################
    # Setup logging...
    getattr(logging, args.loglevel.upper())
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    log_format = '%(asctime)s:%(levelname)s:[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s'
    logging.basicConfig(filename='usb_tld_metrics.log', level=numeric_level, format=log_format)

    print_console_and_log('************************** Application Start **************************')
    print_console_and_log(os.path.basename(__file__) + ' Version {version}'.format(version=__version__))
    print_console_and_log('GEHC_TLD_GENERATION_DATE = ' + tags.GEHC_TLD_GENERATION_DATE)
    logging.info(' '.join(sys.argv))
    print_console_and_log('')

    ###############################################################################
    # Attempt to load the YAML test config file.
    sample_drift_config = read_yaml(args.yaml_config)
    if sample_drift_config is None:
        print("Error reading YAML file " + args.yaml_config )
        return -1

    ###############################################################################
    # Does the user want to collect data?
    if args.collect is not None:
        ###############################################################################
        # Attempt to read yaml collection dictionary.
        G_COLLECTION_CONFIG = read_yaml(G_COLLECTION_FILENAME)
        if G_COLLECTION_CONFIG is None:
            print("Error reading YAML file " + args.collect)
            return -1
        # Does the users collection tag exist?
        if args.collect not in G_COLLECTION_CONFIG:
            print(args.collect + ' not found in collection dictionary')
            return -1

    ###############################################################################
    # Test for existence of the LeCroy file.
    if os.path.isfile(args.csv_input_file) is False:
        print('ERROR, ' + args.csv_input_file + ' does not exist')
        print('\n\n')
        parser.print_help()
        return -1

    ###############################################################################
    # Test for existence of the graphics files extension (if one was given).
    if args.graphics_extension is not None:
        # Check if the given extension is supported.
        supported_extenion_list = list(plt.gcf().canvas.get_supported_filetypes().keys())
        if args.graphics_extension not in supported_extenion_list:
            print('ERROR, unsupported graphics extension : ' + args.graphics_extension)
            print('Supported extensions: ' + str(supported_extenion_list))
            print('\n\n')
            parser.print_help()
            return -1

    ###############################################################################
    # Did the user spec a graphics directory?
    if args.graphics_dir is not None:
        # Test for existence of optional graphics directory.
        if os.path.isdir(args.graphics_dir) is False:
            print('ERROR, ' + args.graphics_dir + ' is not a directory')
            print('\n\n')
            parser.print_help()
            return -1
    else:
        args.graphics_dir = os.getcwd()  # use the current directory if none specified

    ###############################################################################
    # Did the user spec an output CSV file?  If not, we construct one for them.
    if args.csv_output_file is None:
        # A output file name was not given, so build one
        args.csv_output_file = os.path.splitext(os.path.basename(args.csv_input_file))[0] + '_metrics.csv'

    # endregion

    ###############################################################################
    # Read the LeCroy CSV file in...
    print('Reading ' + args.csv_input_file + '...')
    try:
        df = pd.read_csv(args.csv_input_file, usecols=REQUIRED_COLUMNS, converters={'Data': stripspace,
                                                                                    'Time Stamp': stripspace})
    except:
        logging.warning('ERROR encountered reading CSV file %s', args.csv_input_file, exc_info=True)
        print('ERROR encountered reading CSV file: ' + args.csv_input_file)
        print('This file MUST BE a CSV file exported from LeCroy while in spreadsheet view')
        print('\n\n')
        parser.print_help()
        return -1

    if args.verbose is True:
        print(df.info())

    # Did we find all the required columns?
    if set(REQUIRED_COLUMNS).issubset(df.columns.values.tolist()) is False:
        logging.warning('Subset of required columns not present', exc_info=True)
        print('Subset of required columns not present, did you export from LeCroy correctly?')
        print('\tSubset expected ' + str(REQUIRED_COLUMNS))
        print('\tActual ' + str(df.columns.values.tolist()))
        return -1

    ###############################################################################
    # Get rid of anything that is not a transfer or transaction...
    # At the moment we don't care about anything else.
    df = df[(df.Item.str.startswith('Tra')) | (df.Item.str.startswith('Xfr'))]

    ###############################################################################
    # Check minimum row count required.
    if len(df) < MIN_ROW_COUNT:
        print('CSV requires at least ' + str(MIN_ROW_COUNT) + ' rows')
        print(args.csv_input_file + ' had ' + str(len(df)) + ' rows')
        return -1

    ###############################################################################
    # Remove all columns except those with more than 10 real values.
    df = df.dropna(axis=1, how='all', thresh=MIN_ROW_COUNT)

    # Remove rows that have certain columns with no value.
    # Note this is required for subsequent processing.
    df.dropna(subset=['Data', 'Addr,Endp,Dir'], inplace=True)

    # Split this goofy column into 3 columns
    print('Splitting Addr,Endp,Dir column...')
    df['Addr'] = df['Addr,Endp,Dir'].str.split(',').str[0]
    df['Endp'] = df['Addr,Endp,Dir'].str.split(',').str[1]
    df['Dir'] = df['Addr,Endp,Dir'].str.split(',').str[2].str.strip()
    df.drop(['Addr,Endp,Dir'], axis='columns', inplace=True)  # drop it, don't need it no more

    if args.verbose is True:
        print(df.info())  # optionally print metrics

    # Set data type on certain columns
    print('Setting data types...')
    df['Time Stamp'] = pd.to_numeric(df['Time Stamp'])
    df[['Endp', 'Addr']] = df[['Endp', 'Addr']].astype('uint64')
    df['USB_Trans_Count'] = np.nan

    # Get rid of additional rows we don't care about.
    # This would be a Endpoint or Address of zero
    df = df[df.Endp != 0]
    df = df[df.Addr != 0]

    ###############################################################################
    # Does the user want to discard the analog output channel?
    if args.discard_analog is True:
        df = df[df.Endp != 2]       # analog output is on end point to

    ###############################################################################
    # Scope analysis from given start time to end of collection.
    df = df[df['Time Stamp'] >= args.start_time_seconds]

    if args.verbose is True:
        print(df.info())  # optionally print metrics

    # Split into groups based on direction, end point and addr...
    print('Splitting collection into groups...')
    groups = df.groupby(['Dir', 'Endp', 'Addr'])
    logging.info('Groups in collection: ' + str(groups.size()))
    if args.verbose is True:
        print(groups.size())

    # Create sequence column.
    print('Calculating transactions per transfer...')
    for key, g in groups:
        # Create new column with sequential values...
        g['USB_Trans_Count'] = range(len(g))
        # Filter out everything but Transfers...
        xfrs = (g[g.Item.str.startswith('Xfr')]).USB_Trans_Count.diff().shift(-1)
        df.update(xfrs.add(-1))

    if args.verbose is True:
        print(df.info())  # optionally print metrics

    # Combine all transactions in this transfer to single line (for parsing)
    print('Collapsing transaction(s) to transfer(s)...')
    final = pd.DataFrame()
    for key, g in groups:
        g['Time Stamp'] = g['Time Stamp'].shift(-1)  # replace transfer timestamp with 1st transaction timestamp
        g.Data = np.where(pd.isnull(g.USB_Trans_Count), g.Data,
                          np.where(g.USB_Trans_Count == 1, g.Data.shift(-1).str[2:],
                          np.where(g.USB_Trans_Count == 2, g.Data.shift(-1).str[2:] + g.Data.shift(-2).str[2:],
                          np.where(g.USB_Trans_Count == 3,g.Data.shift(-1).str[2:] + g.Data.shift(-2).str[2:] + g.Data.shift(-3).str[2:],
                          np.where(g.USB_Trans_Count == 4,g.Data.shift(-1).str[2:] + g.Data.shift(-2).str[2:] + g.Data.shift(-3).str[2:] + g.Data.shift(-4).str[2:],
                          np.where(g.USB_Trans_Count == 5,g.Data.shift(-1).str[2:] + g.Data.shift(-2).str[2:] + g.Data.shift(-3).str[2:] + g.Data.shift(-4).str[2:] + g.Data.shift(-5).str[2:],np.nan))))))
        final = final.combine_first(g)

    # Attempt to minimize memory footprint
    final = final[final.Item.str.startswith('Xfr')]
    del df  # we don't need the original dataframe anymore

    # Remove Data rows with no data
    logging.info('Count of Data rows with null: %i', final.Data.isnull().sum())
    final.dropna(subset=['Data'], inplace=True)

    # Extract additional metatdata from transfer...
    print('Extracting metadata...')
    final['Device'] = final.Data.apply(to_device)  # get the device by name
    final.dropna(subset=['Device'], inplace=True)  # any row without a device name should be removed

    # Sanity check, do we have any more rows left?
    if len(final) == 0:
        # Something wrong here...
        print('EMPTY DATAFRAME AFTER REMOVING ROWS WITH NO DEVICE NAME, was this file exported correctly?')
        logging.warning('EMPTY DATAFRAME AFTER REMOVING ROWS WITH NO DEVICE NAME')
        return -1

    # Coninute extracing metatdata...
    final[['Endp', 'Addr', 'USB_Trans_Count']] = final[['Endp', 'Addr', 'USB_Trans_Count']].astype('uint64')
    final['TLD Length'] = final.Data.apply(to_payload_length)  # decode tld payload length
    final['TLD Seq'] = final.Data.apply(to_sequence)  # decode tld sequence #
    final['USB Transfer'] = final.Item.apply(to_trans)  # decode tld transfer # (useful to match with lecroy capture)
    final['tags'] = final.apply(tld_to_string, axis=1)  # decode to tags by name
    final.drop(['Item'], axis='columns', inplace=True)  # don't need this column anymore
    if args.verbose is True:
        print(final.info())  # optionally print metrics

    # Get time span...
    time_span = final['Time Stamp'].iloc[-1] - final['Time Stamp'].iloc[0]
    print('Duration of collection is ' + str(timedelta(seconds=time_span)))

    ###############################################################################
    # Process and write statistics.
    # Open CSV file...
    print('Processing statistics...')
    try:
        # Rip through the discovered tags and write stats to dataframe...
        for parent_key, child_count in g_children_found.items():
            parent_tag, endp, direction, child_tag = parent_key.split(':')

            # Display what weez doen
            print('Processing ' + parent_tag + '----' + child_tag + '...')

            # Calculate time between each occurrence
            delta = (final[final.tags.str.contains(str(parent_key + ','))])['Time Stamp'].iloc[
                    1:].diff()  # calculate time difference per row

            # Fill the series with our calculated values
            g_metrics_series['Parent Tag'] = parent_tag
            g_metrics_series['Child Tag'] = child_tag
            g_metrics_series['Endp'] = endp
            g_metrics_series['Dir'] = direction
            g_metrics_series['Child Tag Count'] = child_count
            g_metrics_series['Child Length Total'] = g_child_payload_tally[parent_key]
            g_metrics_series['Ratio'] = g_child_payload_tally[parent_key] / float(child_count)
            g_metrics_series['Mean (mS)'] = delta.mean() * 1000
            g_metrics_series['Median (mS)'] = delta.median() * 1000
            g_metrics_series['Max (mS)'] = delta.max() * 1000
            g_metrics_series['Min (mS)'] = delta.min() * 1000
            g_metrics_series['Std'] = delta.std() * 1000
            g_metrics_series['Mean Child SPS'] = 1 / delta.mean()
            g_metrics_series['Max Child SPS'] = 1 / delta.min()
            g_metrics_series['Min Child SPS'] = 1 / delta.max()
            g_metrics_series['Sample Drift Measured MAX ppm'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured MAX Sec'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured MIN ppm'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured MIN Sec'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured MEAN'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured STD'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Expected seconds'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Samples Collected'] = G_NOT_APPLICABLE

            # Is this parent/child tag pair in our test dictionary?
            if parent_tag in sample_drift_config and child_tag in sample_drift_config[parent_tag]:
                needed_samples = sample_drift_config[parent_tag][child_tag]['samples']
                expected_seconds = sample_drift_config[parent_tag][child_tag]['seconds']
                # Do we have enough samples to calculate sample drift?
                if child_count > needed_samples:
                    delta_series = (final[final.tags.str.contains(str(parent_key + ','))])['Time Stamp'].iloc[1:].diff(
                        needed_samples)
                    g_metrics_series['Sample Drift Measured MAX ppm'] = (delta_series.max()-expected_seconds) / \
                                                                        expected_seconds * 1e6
                    g_metrics_series['Sample Drift Measured MAX Sec'] = delta_series.max()
                    g_metrics_series['Sample Drift Measured MIN ppm'] = (delta_series.min()-expected_seconds) / \
                                                                        expected_seconds * 1e6
                    g_metrics_series['Sample Drift Measured MIN Sec'] = delta_series.min()
                    g_metrics_series['Sample Drift Measured MEAN'] = delta_series.mean()
                    g_metrics_series['Sample Drift Measured STD'] = delta_series.std()
                    g_metrics_series['Sample Drift Expected seconds'] = expected_seconds
                    g_metrics_series['Sample Drift Samples Collected'] = delta_series.count()
                else:
                    # Tell user there's not enough samples to test sample drift on.
                    print_console_and_log(
                        '\n\t*** Not enough samples to perform sample drift calculation for: ' +
                        parent_tag + '----' + child_tag + '\n')
                    g_metrics_series['Sample Drift Measured MAX ppm'] = G_NOT_ENOUGH_SAMPLES
                    g_metrics_series['Sample Drift Measured MAX Sec'] = G_NOT_ENOUGH_SAMPLES
                    g_metrics_series['Sample Drift Measured MIN ppm'] = G_NOT_ENOUGH_SAMPLES
                    g_metrics_series['Sample Drift Measured MIN Sec'] = G_NOT_ENOUGH_SAMPLES
                    g_metrics_series['Sample Drift Measured MEAN'] = G_NOT_ENOUGH_SAMPLES
                    g_metrics_series['Sample Drift Measured STD'] = G_NOT_ENOUGH_SAMPLES
                    g_metrics_series['Sample Drift Expected seconds'] = G_NOT_ENOUGH_SAMPLES
                    g_metrics_series['Sample Drift Samples Collected'] = G_NOT_ENOUGH_SAMPLES

            # Tack this just built metrics row on the end of the dataframe.
            g_metrics_df = g_metrics_df.append(g_metrics_series, ignore_index=True)

            # Check if this is a surface temperature response...
            # We need to capture certain metrics for use when we calc temp response.
            if parent_tag == 'ACTIVE_CABLE_CP_CONTROL' and child_tag == 'ACTIVE_CABLE_TEMPERATURE':
                # Store away metrics for use in response time calculation
                surface_temp_child_count = child_count
                surface_temp_endp = endp
                surface_temp_child_pay_load_tally = g_child_payload_tally[parent_key]

            # Should we create a graphic for this one?
            if args.graphics_extension is not None and delta.size > 1:
                print('Plotting ' + args.graphics_dir + '\\' + parent_tag + '--' + endp + '--' + direction +
                      '--' + child_tag + '.' + args.graphics_extension)
                p = delta.plot(kind='hist', title=parent_key)
                plt.savefig(os.path.join(args.graphics_dir, parent_tag + '--' + endp + '--' + direction +
                                         '--' + child_tag + '.' + args.graphics_extension))
                plt.close(p.get_figure())

        # Calculate the surface temperature response times...
        delta = get_temp_delta(final)

        # Do we have any response times from the calculation?
        if delta is not None:
            # Ok, proceed with histrogram creation...
            g_metrics_series['Parent Tag'] = 'Temp Response Time'
            g_metrics_series['Child Tag'] = 'Request/Reponse'
            g_metrics_series['Endp'] = surface_temp_endp
            g_metrics_series['Dir'] = 'OUT to IN'
            g_metrics_series['Child Tag Count'] = surface_temp_child_count
            g_metrics_series['Child Length Total'] = surface_temp_child_pay_load_tally
            g_metrics_series['Ratio'] = surface_temp_child_pay_load_tally / float(surface_temp_child_count)
            g_metrics_series['Mean (mS)'] = delta.mean() * 1000
            g_metrics_series['Median (mS)'] = delta.median() * 1000
            g_metrics_series['Max (mS)'] = delta.max() * 1000
            g_metrics_series['Min (mS)'] = delta.min() * 1000
            g_metrics_series['Std'] = delta.std() * 1000
            g_metrics_series['Mean Child SPS'] = G_NOT_APPLICABLE
            g_metrics_series['Max Child SPS'] = G_NOT_APPLICABLE
            g_metrics_series['Min Child SPS'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured MAX ppm'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured MAX Sec'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured MIN ppm'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured MIN Sec'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured MEAN'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Measured STD'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Expected seconds'] = G_NOT_APPLICABLE
            g_metrics_series['Sample Drift Samples Collected'] = G_NOT_APPLICABLE
            g_metrics_df = g_metrics_df.append(g_metrics_series, ignore_index=True)

            # Should we create a graphic for this?
            if args.graphics_extension is not None and delta.size > 1:
                # Ok, proceed with histogram creation...
                p = delta.plot(kind='hist', title='Temp Request Response')
                plt.savefig(os.path.join(args.graphics_dir, 'temp_response' + '.' + args.graphics_extension))
                plt.close(p.get_figure())

        else:
            print('Could not create temperature response histogram.')

    except:
        print('Error processing statistics')
        logging.warning('Error processing and writing statistics', exc_info=True)
        return -1

    ###############################################################################
    # Write the metrics file.
    print('Writing metrics to ' + args.csv_output_file + '...')
    g_metrics_df[g_metrics_df_int_value_list] = g_metrics_df[g_metrics_df_int_value_list].astype('uint64')
    g_metrics_df[g_metrics_df_float_value_list] = g_metrics_df[g_metrics_df_float_value_list].astype('float64')
    # Ok, let's sort and write to CSV file.
    g_metrics_df.sort_values(by=['Child Tag', 'Parent Tag', 'Dir', 'Child Tag Count', 'Endp'],
                             ascending=[True, True, True, False, False],
                             inplace=True)

    g_metrics_df.to_csv(args.csv_output_file, index=False, float_format='%.{0}f'.format(args.precision))

    ###############################################################################
    # Do we need to write the data csv?
    # Recall this dataframe was used to build the metrics file and is useful for further analysis.
    if args.store_csv_data_file is True:
        # Note the data file is written to the same directory as the output file.
        csv_data_filename = os.path.splitext(args.csv_output_file)[0] + '_data.csv'
        print('Writing dataframe to ' + csv_data_filename + '...')
        final.to_csv(csv_data_filename, index=False, float_format='%.9f')

    ###############################################################################
    # Do we have child data to find and log?
    if args.collect is not None and G_COLLECTION_CONFIG is not None:
        if args.collect != 'ANALOG_OUTPUT':
            print('Extracting ' + args.collect + ' metrics and writing to file')
            # Looks like we gotsa collection to log.
            collection_config = G_COLLECTION_CONFIG[args.collect]
            write_colleciton(final, collection_config['tags'], collection_config['datatype'], args.collect + '.csv')
        else:
            '''
            Debug
            test_ao_wf_series = to_series(final, 122, 1)   # get a test version to determine sample count
            '''
            ao_wf_series = to_series(final, 122, 1, True)     # get the analog output waveform as a series with counts
            ao_wf_series = ao_wf_series.astype(dtype='int16')   # set the correct data type.
            ao_marker_series = to_series(final, 122, 2)         # get the analog output markers as a series

            # Process the series...
            ao_wf_series_index = 0
            series_to_build = pd.Series(dtype='int16')   # series to build
            for index, marker_flag in ao_marker_series.iteritems():
                data_length = ao_wf_series.iloc[ao_wf_series_index]
                assert data_length == 4 or data_length == 5     # must be 4 or 5
                # QRS and PACE are indicated in the marker flag.  Zero mean neither are present in this sample set.
                if marker_flag & QRS_COMPLEX_DETECTED_MASK is not 0 and 0:
                    # We have a QRS detected within the sample set.
                    # Get the sample number on which it occurred.
                    qrs_sample_index = (marker_flag & QRS_SAMPLE_INDEX_MASK) >> QRS_SAMPLE_INDEX_SHIFT

                    # Adjust index if needed.  This is due to current implementation of URE2's beat detector...
                    if qrs_sample_index == 4 and data_length == 4:
                        qrs_sample_index = 3

                    assert qrs_sample_index <= 4    # sanity check value
                    ao_wf_series.iloc[ao_wf_series_index+1+qrs_sample_index] += QRS_MARKER_OFFSET   # and mark

                # Copy over wf samples to series we are building...
                series_to_build = \
                    series_to_build.append(ao_wf_series[ao_wf_series_index+1:ao_wf_series_index+1+data_length])
                # Advance to next child count...
                ao_wf_series_index += (data_length + 1)

            '''
            # Debug
            print('AO wf sample length: {0:d}, AO wf sample length with counts inserted: {1:d}, AO marker length: {2:d},
             ao_wf_series_index: {3:d}'.format(len(test_ao_wf_series), len(ao_wf_series), len(ao_marker_series), 
             ao_wf_series_index))
            '''

            print('Writing analog waveform array to ao_wf.csv')
            series_to_build.to_csv('ao_wf.csv', index=False)

    ###############################################################################
    # Did the user want a COMMS_DATA output?
    if args.comms_data is True:
        # Construct filename...
        c_filename = os.path.splitext(args.csv_output_file)[0] + '_array.c'
        print('Writing COMMS_DATA C array to ' + c_filename)
        if write_comms_data(final, c_filename) is False:
            print_console_and_log('Failure writing COMMS_DATAQ C array to ' + c_filename)

    ###############################################################################
    # Test sequence number of each end point direction and device.
    print('Splitting into groups based on Device, direction, end point and address')
    final_groups = final.groupby(['Device', 'Dir', 'Endp', 'Addr'])
    for key, g in final_groups:
        # Create tuple string...
        tuple_string = '{0:s}_{1:s}_{2:d}_{3:d}'.format(key[0], key[1], key[2], key[3])
        print('\n')
        print('Testing TLD sequence for {0:s}'.format(tuple_string))
        previous_sequence = START_TLD_SEQUENCE  # will fail first test causing adoption of first seq number
        # Iterate though entire frame and examine/test the TLD sequence numbers...
        for index, row in g.iterrows():
            if row['TLD Seq'] != previous_sequence:
                # Is this really a failure?
                if previous_sequence != START_TLD_SEQUENCE:
                    print('FAIL, TLD sequence skip detected, expected: {0:d}, observed: {1:d}.'
                          .format(previous_sequence, row['TLD Seq']))
                # Adapt new sequence
                previous_sequence = row['TLD Seq']
            # Next sequence...
            previous_sequence = previous_sequence + 1
            previous_sequence = previous_sequence & 0xffff  # wrap
        # Did the user want to save the group?
        if args.separate_streams is True:
            # Build filename for this group.
            stream_filename = os.path.splitext(args.csv_output_file)[0] + '_' + tuple_string + '_data.csv'
            print('Writing: {0:s}...'.format(stream_filename))
            g.to_csv(stream_filename, index=False, float_format='%.9f')

    print('--- DONE ---')
    return 0


# endregion

###############################################################################
if __name__ == '__main__':
    rv = main()
    exit(rv)
