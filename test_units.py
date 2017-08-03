import os
import pytest
import hashlib
import yaml

# Known good/bad TLDs
G_URE_TLD_1 = '5EEDDECA004B41DB5F2A007A000900010004001700180019001B0002000100400079003900010004FFE1FFE0FFDEFFDD00020004\
               002000220023002500030004003F0041004400470004000400330035003800390005000400440047004C004F00060004002D002F\
               0032003300070004000A000C000E000F000800040009000B000E000F000900040003000400060006002100010040007B00030001\
               0001012AEA5F'
G_URE_TLD_2 = '5EEDDECA000B41D872E4007A000900010004000B000C000D000E0002000100404930'
G_CP_TLD = '5EEDBADE0004013F0A350046000200020000562A'
G_CP_TLD_MALFORMED = '5EEDBADE0004013F0A350046000200020001562A'


# Ensure we properly strip white space from the passed string
def test_stripwhite():
    from usb_tld_metrics import stripspace
    test_string = 'now is the time'
    s = stripspace(test_string)
    print('Testing white space stripping...')
    assert (' ' not in s)
    assert (test_string != s)
    assert (s == 'nowisthetime')


# Test that we translate a TLD into a device string
def test_to_device():
    from usb_tld_metrics import to_device
    print('Testing TLD device id parsing...')
    assert (to_device('5EEDBADE0005013E2D24004600030003000100B7D30D') == 'ACTIVE_CABLE_CP')
    assert (to_device(G_URE_TLD_2) == 'ACTIVE_CABLE_ECG_DSP')
    assert (to_device('5EEDDECA') == 'ACTIVE_CABLE_ECG_DSP')
    print('Testing malformed device id')
    assert (to_device('5EEDDEC') is None)
    assert (to_device('') is None)
    assert (to_device('5EEDDOG') is None)


# Test that we extract the correct sequence number from a tld
def test_to_sequence():
    from usb_tld_metrics import to_sequence
    print('Testing TLD sequence number parsing...')
    assert (to_sequence(G_URE_TLD_1) == 16859)
    assert (to_sequence('5EEDBADE0004013F0A350046000200020000562A') == 319)
    assert (to_sequence('5EEDBADE00040138') == 312)
    print('Testing malformed sequence number')
    assert (to_sequence('') == 0)


# Test for parsing of transaction value
def test_to_trans():
    from usb_tld_metrics import to_trans
    print('Testing USB transaction number parsing...')
    assert (to_trans('Xfr 61') == 61)
    print('Testing malformed USB transaction number parsing...')
    with pytest.raises(IndexError) as excinfo:
        to_trans('Xfr266')
    assert 'IndexError' in str(excinfo.type)
    assert (to_trans('Tra 61') == 0)
    assert (to_trans('fish 23') == 0)


# Test for correct parsing out of tags.
# Note the tld_tags is a generator
def test_tld_tags():
    from usb_tld_metrics import tld_tags
    import tags

    # First iteration will be a list that we will test.
    print('Testing ' + G_CP_TLD)
    g = tld_tags(G_CP_TLD)
    tags_list = next(g)
    assert (len(tags_list) == 3)
    assert tags.DeviceId_Dict[tags_list[0]] == 'ACTIVE_CABLE_CP'
    assert tags.ParentTag_Dict[tags_list[1]] == 'ACTIVE_CABLE_CP_CONTROL'
    child_dict = tags.ChildTag_Dict[tags_list[1]]
    assert child_dict[tags_list[2][0]] == 'ACTIVE_CABLE_TEMPERATURE_REQUEST'
    assert tags_list[2][1] == 0  # child length should be zero

    # Second iteration should generate a StopIteration exception, cause we are done iterating.
    with pytest.raises(StopIteration) as excinfo:
        next(g)
    assert 'StopIteration' in str(excinfo.type)

    # This is a malformed TLD.  So, first iteration should generate a StopIteration exception
    print('Testing malformed ' + G_CP_TLD_MALFORMED)
    g = tld_tags(G_CP_TLD_MALFORMED)
    with pytest.raises(StopIteration) as excinfo:
        next(g)
    assert 'StopIteration' in str(excinfo.type)

    # Carescape ECG
    print('Testing ' + G_URE_TLD_2)
    g = tld_tags(G_URE_TLD_2)
    tags_list = next(g)
    assert (len(tags_list) == 4)
    assert tags.DeviceId_Dict[tags_list[0]] == 'ACTIVE_CABLE_ECG_DSP'
    assert tags.ParentTag_Dict[tags_list[1]] == 'ACTIVE_CABLE_AO_ECG'
    child_dict = tags.ChildTag_Dict[tags_list[1]]
    assert child_dict[tags_list[2][0]] == 'LEAD'
    assert child_dict[tags_list[3][0]] == 'MARKER_FLAG'
    assert tags_list[2][1] == 4
    assert tags_list[3][1] == 1

    # Second iteration should generate a StopIteration exception, cause we are done iterating.
    with pytest.raises(StopIteration) as excinfo:
        next(g)
    assert 'StopIteration' in str(excinfo.type)

def test_main(tmpdir,lecroy_dir,test):
    """
    Main function test.


    """
    from usb_tld_metrics import main

    # Fail if the dir does not exist
    check_dir(lecroy_dir)

    # Preface file name with the given LeCroy directory
    test['command_arguments'][1] = lecroy_dir + '/' + test['command_arguments'][1]

    # Fail if the file does not exist
    check_file(test['command_arguments'][1])

    # Parse out basename of input file.
    lecroy_exported_file = os.path.splitext(os.path.basename(test['command_arguments'][1]))[0]

    # Overwrite output file string entry with a temporary file.
    metrics_file = tmpdir.join(lecroy_exported_file+'_metrics.csv').__str__()
    data_file = tmpdir.join(lecroy_exported_file+'_metrics_data.csv').__str__()
    test['command_arguments'][-1] = metrics_file

    # Test!
    assert main(test['command_arguments']) == 0     # we expect the command to go off without a hitch
    assert os.path.isfile(metrics_file)             # we expect a metrics file to be created
    assert os.path.isfile(data_file)                # we expect a data file to be created

    # Checksum and compare the output files (metric and data files)
    dos2unix(metrics_file)  # gotta do this cause i haven't figured out how to deal with CRLF
    dos2unix(data_file)
    assert md5sum(metrics_file) == test['metrics_md5sum']
    assert md5sum(data_file) == test['data_md5sum']

def md5sum(filename):
    """
    Support function.
    Calculate the md5sum of the given file.

    Args:
        filename: the filename to calc md5sum on

    Returns:
        the calculated md5sum of the data contained int the file

    """
    blocksize = 65536
    hasher = hashlib.md5()
    with open(filename, 'rb') as afile:
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
    return hasher.hexdigest()


def dos2unix(filename):
    """
    Support function.
    Remove any found carriage returns from the give file.

    Args:
        filename: filename to open remove carriage returns and write back to same file.

    Returns:
        nothing

    """
    text = open(filename, 'rb').read().replace(b'\r\n', b'\n')
    open(filename, 'wb').write(text)


def check_dir(directory_name):
    """
    Support function
    Check if the given directory exists.
    
    Args:
        directory_name (): name of the directory to check

    Returns:
        True - the directory exists
        False - the directory does not exist

    """
    if os.path.isdir(directory_name) is False:
        pytest.fail("Directory {0} does not exist".format(directory_name))
        return False
    else:
        return True


def check_file(file_name):
    """
    Support function
    Check if the given file exists.

    Args:
        file_name (): name of the file to check

    Returns:
        True - the file exists
        False - the file does not exist

    """
    if os.path.isfile(file_name) is False:
        pytest.fail("File {0} does not exist".format(file_name))
        return False
    else:
        return True

