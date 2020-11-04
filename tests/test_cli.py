import os
import pytest
import numpy as np
from numpy.testing import assert_allclose

import emg3d
from emg3d import cli

# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None


@pytest.mark.script_launch_mode('subprocess')
def test_basic(script_runner):

    os.environ["NUMBA_DISABLE_JIT"] = "1"

    # Test the installed version runs by -h.
    ret = script_runner.run('emg3d', '-h')
    assert ret.success
    assert "emg3d is a multigrid solver for 3D EM diffusion" in ret.stdout
    assert ret.stderr == ''

    # Test the installed version fails if called without anything.
    ret = script_runner.run('emg3d')
    assert not ret.success

    # Test emg3d/__main__.py by calling the folder emg3d.
    ret = script_runner.run('python', 'emg3d', '--report')
    assert ret.success
    # Exclude time to avoid errors.
    assert emg3d.utils.Report().__repr__()[115:] in ret.stdout

    # Test emg3d/cli/_main_.py by calling the file - I.
    ret = script_runner.run('python', 'emg3d/cli/main.py', '--version')
    assert ret.success
    assert emg3d.utils.__version__ in ret.stdout

    # Test emg3d/cli/_main_.py by calling the file - II.
    ret = script_runner.run('python', 'emg3d/cli/main.py', '--report')
    assert ret.success
    # Exclude time to avoid errors.
    assert emg3d.utils.Report().__repr__()[115:] in ret.stdout

    # Test emg3d/cli/_main_.py by calling the file - III.
    ret = script_runner.run('python', 'emg3d/cli/main.py', '-d')
    assert not ret.success
    assert "CONFIGURATION FILE NOT FOUND" in ret.stderr

    os.environ["NUMBA_DISABLE_JIT"] = "0"


class TestParser:

    # Default terminal values
    args_dict = {
            'config': 'emg3d.cfg',
            'nproc': None,
            'forward': False,
            'misfit': False,
            'gradient': False,
            'path': None,
            'survey': None,
            'model': None,
            'output': None,
            'verbosity': 0,
            'dry_run': False,
            }

    def test_term_config(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[files]\n")
            f.write(f"path={tmpdir}")

        # Name provided.
        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert config == term['config_file']

        # Check some default values.
        assert term['function'] == 'forward'
        assert cfg['files']['survey'] == tmpdir+'/survey.h5'
        assert cfg['files']['model'] == tmpdir+'/model.h5'
        assert cfg['files']['output'] == tmpdir+'/emg3d_out.h5'
        assert cfg['files']['log'] == tmpdir+'/emg3d_out.log'

        # Provide file names
        args_dict = self.args_dict.copy()
        args_dict['survey'] = 'test.h5'
        args_dict['model'] = 'unkno.wn'
        args_dict['output'] = 'out.npz'
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert cfg['files']['survey'] == tmpdir+'/test.h5'
        assert cfg['files']['model'] == tmpdir+'/unkno.h5'
        assert cfg['files']['output'] == tmpdir+'/out.npz'

        # .-trick.
        args_dict = self.args_dict.copy()
        args_dict['config'] = '.'
        _, term = cli.parser.parse_config_file(args_dict)
        assert term['config_file'] == '.'

        # Not existent.
        args_dict = self.args_dict.copy()
        args_dict['config'] = 'bla'
        _, term = cli.parser.parse_config_file(args_dict)
        assert term['config_file'] is False

    def test_term_various(self, tmpdir):

        args_dict = self.args_dict.copy()
        args_dict['nproc'] = -1
        args_dict['verbosity'] = 20
        args_dict['dry_run'] = True
        args_dict['gradient'] = True
        args_dict['path'] = tmpdir
        args_dict['survey'] = 'testit'
        args_dict['model'] = 'model.json'
        args_dict['output'] = 'output.npz'
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert term['verbosity'] == 2  # Maximum 2!
        assert term['dry_run'] is True
        assert term['function'] == 'gradient'
        assert cfg['simulation_options']['max_workers'] == 1
        assert cfg['files']['survey'] == tmpdir+'/testit.h5'
        assert cfg['files']['model'] == tmpdir+'/model.json'
        assert cfg['files']['output'] == tmpdir+'/output.npz'
        assert cfg['files']['log'] == tmpdir+'/output.log'

        with pytest.raises(TypeError, match="Unexpected parameter in"):
            args_dict = self.args_dict.copy()
            args_dict['unknown'] = True
            _ = cli.parser.parse_config_file(args_dict)

    def test_files(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[files]\n")
            f.write(f"path={tmpdir}\n")
            f.write("survey=testit.json\n")
            f.write("model=thismodel\n")
            f.write("output=results.npz\n")
            f.write("store_simulation=false")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert cfg['files']['survey'] == tmpdir+'/testit.json'
        assert cfg['files']['model'] == tmpdir+'/thismodel.h5'
        assert cfg['files']['output'] == tmpdir+'/results.npz'
        assert cfg['files']['log'] == tmpdir+'/results.log'
        assert cfg['files']['store_simulation'] is False

        with pytest.raises(TypeError, match="Unexpected parameter in"):
            # Write a config file.
            config = os.path.join(tmpdir, 'emg3d.cfg')
            with open(config, 'w') as f:
                f.write("[files]\n")
                f.write(f"path={tmpdir}\n")
                f.write("whatever=bla")
            args_dict = self.args_dict.copy()
            args_dict['config'] = config
            cfg, term = cli.parser.parse_config_file(args_dict)

    def test_simulation(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[simulation]\n")
            f.write("max_workers=5\n")
            f.write("gridding=fancything\n")
            f.write("name=PyTest simulation\n")
            f.write("min_offset=1320")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert cfg['simulation_options']['max_workers'] == 5
        assert cfg['simulation_options']['gridding'] == 'fancything'
        assert cfg['simulation_options']['name'] == "PyTest simulation"
        assert cfg['simulation_options']['min_offset'] == 1320.0

        with pytest.raises(TypeError, match="Unexpected parameter in"):
            with open(config, 'a') as f:
                f.write("\nanother=True")
            args_dict = self.args_dict.copy()
            args_dict['config'] = config
            _ = cli.parser.parse_config_file(args_dict)

    def test_solver(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[solver_opts]\n")
            f.write("sslsolver=False\n")
            f.write("cycle=V\n")
            f.write("tol=1e-4\n")
            f.write("nu_init=2")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        test = cfg['simulation_options']['solver_opts']
        assert test['sslsolver'] is False
        assert test['cycle'] == 'V'
        assert test['tol'] == 0.0001
        assert test['nu_init'] == 2

        with pytest.raises(TypeError, match="Unexpected parameter in"):
            with open(config, 'a') as f:
                f.write("\nanother=True")
            args_dict = self.args_dict.copy()
            args_dict['config'] = config
            _ = cli.parser.parse_config_file(args_dict)

    def test_data(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[data]\n")
            f.write("sources=Tx11\n")
            f.write("receivers=Rx1, Rx2\n")
            f.write("frequencies=1")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        test = cfg['data']
        assert test['sources'] == ['Tx11']
        assert test['receivers'] == ['Rx1', 'Rx2']
        assert test['frequencies'] == ['1']

        with pytest.raises(TypeError, match="Unexpected parameter in"):
            # Write a config file.
            config = os.path.join(tmpdir, 'emg3d.cfg')
            with open(config, 'w') as f:
                f.write("[data]\n")
                f.write("whatever=bla")
            args_dict = self.args_dict.copy()
            args_dict['config'] = config
            cfg, term = cli.parser.parse_config_file(args_dict)

    def test_gridding(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[gridding_opts]\n")
            f.write("verb=3\n")
            f.write("frequency=7.77\n")
            f.write("seasurface=-200\n")
            f.write("max_buffer=5000\n")
            f.write("lambda_factor=0.8\n")
            f.write("mapping=LnResistivity\n")
            f.write("vector=yz\n")
            f.write("domain=-2000, 2000; None; -4000, 1.111\n")
            f.write("stretching=1.1, 2.0\n")
            f.write("min_width_limits=20, 40; 30, 60; None\n")
            f.write("properties=0.3, 1, 2, 3, 4, 5, 6\n")
            f.write("center=0, 0, -500\n")
            f.write("cell_number=20, 40, 80, 100\n")
            f.write("min_width_pps=2, 3, 4\n")
            f.write("expand=1, 2")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)

        check = cfg['simulation_options']['gridding_opts']

        assert check['verb'] == 3
        assert check['frequency'] == 7.77
        assert check['seasurface'] == -200
        assert check['max_buffer'] == 5000
        assert check['lambda_factor'] == 0.8
        assert check['mapping'] == 'LnResistivity'
        assert check['vector'] == 'yz'
        assert check['domain'] == ([-2000.0, 2000.0], None, [-4000.0, 1.111])
        assert check['stretching'] == [1.1, 2.0]
        assert check['min_width_limits'] == ([20, 40], [30, 60], None)
        assert check['properties'] == [0.3, 1, 2, 3, 4, 5, 6]
        assert check['center'] == [0, 0, -500]
        assert check['cell_number'] == [20, 40, 80, 100]
        assert check['min_width_pps'] == [2, 3, 4]
        assert check['expand'] == [1, 2]

        with pytest.raises(TypeError, match="Unexpected parameter in"):
            with open(config, 'a') as f:
                f.write("\nanother=True")
            args_dict = self.args_dict.copy()
            args_dict['config'] = config
            _ = cli.parser.parse_config_file(args_dict)


@pytest.mark.skipif(xarray is None, reason="xarray not installed.")
class TestRun:

    # Default values for run-tests
    args_dict = {
            'config': '.',
            'nproc': 1,
            'forward': False,
            'misfit': False,
            'gradient': True,
            'path': None,
            'survey': 'survey.npz',
            'model': 'model.npz',
            'output': 'output.npz',
            'verbosity': 0,
            'dry_run': True,
            }

    if xarray is not None:
        # Create a tiny dummy survey.
        data = np.ones((1, 17, 1))
        data[0, 8:11, 0] = np.nan
        survey = emg3d.Survey(
            name='CLI Survey',
            sources=(4125, 4000, 4000, 0, 0),
            receivers=(np.arange(17)*250+2000, 4000, 3950, 0, 0),
            frequencies=1,
            noise_floor=1e-15,
            relative_error=0.05,
            data=data,
        )

    # Create a dummy grid and model.
    xx = np.ones(16)*500
    grid = emg3d.TensorMesh([xx, xx, xx], x0=np.array([0, 0, 0]))
    model = emg3d.Model(grid, 1.)

    def test_basic(self, tmpdir, capsys):

        # Store survey and model.
        self.survey.to_file(os.path.join(tmpdir, 'survey.npz'), verb=0)
        emg3d.save(os.path.join(tmpdir, 'model.npz'), model=self.model,
                   mesh=self.grid, verb=0)

        args_dict = self.args_dict.copy()
        args_dict['path'] = tmpdir
        args_dict['verbosity'] = -1
        cli.run.simulation(args_dict)

        args_dict = self.args_dict.copy()
        args_dict['path'] = tmpdir
        args_dict['config'] = 'bla'
        args_dict['verbosity'] = 2
        _, _ = capsys.readouterr()
        cli.run.simulation(args_dict)
        _, outstr = capsys.readouterr()
        assert "* WARNING :: CONFIGURATION FILE NOT FOUND." in outstr

    def test_run(self, tmpdir, capsys):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[files]\n")
            f.write("store_simulation=True\n")
            f.write("[solver_opts]\n")
            f.write("sslsolver=False\n")
            f.write("semicoarsening=False\n")
            f.write("linerelaxation=False\n")
            f.write("maxit=1\n")

        # Store survey and model.
        self.survey.to_file(os.path.join(tmpdir, 'survey.npz'), verb=1)
        emg3d.save(os.path.join(tmpdir, 'model.npz'), model=self.model,
                   mesh=self.grid, verb=1)

        # Run a dry run (to output.npz).
        args_dict = self.args_dict.copy()
        args_dict['config'] = os.path.join(tmpdir, 'emg3d.cfg')
        args_dict['path'] = tmpdir
        cli.run.simulation(args_dict)

        # Actually run one iteration (to output2.npz).
        args_dict = self.args_dict.copy()
        args_dict['config'] = os.path.join(tmpdir, 'emg3d.cfg')
        args_dict['path'] = tmpdir
        args_dict['dry_run'] = False
        args_dict['output'] = 'output2.npz'
        cli.run.simulation(args_dict)

        # Ensure dry_run returns same shaped data as the real thing.
        res1 = emg3d.load(os.path.join(tmpdir, 'output.npz'))
        res2 = emg3d.load(os.path.join(tmpdir, 'output2.npz'))
        assert_allclose(res1['data'].shape, res2['data'].shape)
        assert_allclose(res1['misfit'].shape, res2['misfit'].shape)
        assert_allclose(res1['gradient'].shape, res2['gradient'].shape)
        assert 'simulation' in res2
        assert res1['n_observations'] == np.isfinite(self.data).sum()
        assert res2['n_observations'] == np.isfinite(self.data).sum()

        # Actually run one iteration (to output2.npz).
        args_dict = self.args_dict.copy()
        args_dict['config'] = os.path.join(tmpdir, 'emg3d.cfg')
        args_dict['path'] = tmpdir
        args_dict['forward'] = True
        args_dict['gradient'] = False
        args_dict['dry_run'] = False
        args_dict['output'] = 'output3.npz'
        cli.run.simulation(args_dict)
        res3 = emg3d.load(os.path.join(tmpdir, 'output3.npz'))
        assert 'misfit' not in res3
        assert 'gradient' not in res3

    def test_data(self, tmpdir, capsys):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[data]\n")
            f.write("sources=Tx0\n")
            f.write("receivers=Rx04, Rx09, Rx01, Rx11, Rx05\n")
            f.write("frequencies=1")

        # Store survey and model.
        self.survey.to_file(os.path.join(tmpdir, 'survey.npz'), verb=1)
        emg3d.save(os.path.join(tmpdir, 'model.npz'), model=self.model,
                   mesh=self.grid, verb=1)

        # Run a dry run (to output.npz).
        args_dict = self.args_dict.copy()
        args_dict['config'] = os.path.join(tmpdir, 'emg3d.cfg')
        args_dict['path'] = tmpdir
        cli.run.simulation(args_dict)

        # Ensure dry_run returns same shaped data as the real thing.
        res = emg3d.load(os.path.join(tmpdir, 'output.npz'))
        assert_allclose(res['data'].shape, (1, 5, 1))
        assert res['n_observations'] == 4
