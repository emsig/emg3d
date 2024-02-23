import os
import sys
import pytest
import numpy as np
from os.path import join, sep
from numpy.testing import assert_allclose
from contextlib import suppress, ContextDecorator

import emg3d
from emg3d import cli

# Soft dependencies
try:
    import xarray
except ImportError:
    xarray = None


class disable_numba(ContextDecorator):
    """Context decorator to disable-enable JIT and remove log file."""
    def __enter__(self):
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        return self

    def __exit__(self, *exc):
        os.environ["NUMBA_DISABLE_JIT"] = "0"
        # Clean up
        with suppress(FileNotFoundError):
            os.remove('emg3d_out.log')
        return False


@disable_numba()
@pytest.mark.script_launch_mode('subprocess')
def test_main(script_runner):

    # Test the installed version runs by -h.
    ret = script_runner.run(['emg3d', '-h'])
    assert ret.success
    assert "Multigrid solver for 3D electromagnetic diffusion." in ret.stdout

    # Test the info printed if called without anything.
    ret = script_runner.run('emg3d')
    assert ret.success
    assert "Multigrid solver for 3D electromagnetic diffusion." in ret.stdout
    assert "emg3d v" in ret.stdout

    # Test emg3d/cli/_main_.py by calling the file - I.
    ret = script_runner.run(
            ['python', join('emg3d', 'cli', 'main.py'), '--version'])
    assert ret.success
    assert "emg3d v" in ret.stdout

    # Test emg3d/cli/_main_.py by calling the file - II.
    ret = script_runner.run(
            ['python', join('emg3d', 'cli', 'main.py'), '--report'])
    assert ret.success
    # Exclude time to avoid errors.
    assert emg3d.utils.Report().__repr__()[115:475] in ret.stdout

    # Test emg3d/cli/_main_.py by calling the file - III.
    ret = script_runner.run(['python', join('emg3d', 'cli', 'main.py'), '-d'])
    assert not ret.success
    assert "* ERROR   :: Config file not found: " in ret.stderr


# Skip test for minimal, it fails. Not sure why.
# Something multiprocessing/numba/sem_open/synchronization primitives.
@disable_numba()
@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.skipif(sys.version_info < (3, 9), reason="Exclude for Python 3.8")
def test_main2(script_runner):

    # Test emg3d/__main__.py by calling the folder emg3d.
    ret = script_runner.run(['python', 'emg3d', '--report'])
    assert ret.success
    # Exclude time to avoid errors.
    # Exclude empymod-version (after 475), because if run locally without
    # having emg3d installed it will be "unknown" for the __main__ one.
    assert emg3d.utils.Report().__repr__()[115:475] in ret.stdout


class TestParser:

    # Default terminal values
    args_dict = {
            'config': 'emg3d.cfg',
            'nproc': None,
            'forward': False,
            'misfit': False,
            'gradient': False,
            'layered': False,
            'path': None,
            'survey': None,
            'model': None,
            'output': None,
            'save': None,
            'load': None,
            'cache': None,
            'clean': False,
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
        assert cfg['files']['survey'] == join(tmpdir, 'survey.h5')
        assert cfg['files']['model'] == join(tmpdir, 'model.h5')
        assert cfg['files']['output'] == join(tmpdir, 'emg3d_out.h5')
        assert cfg['files']['log'] == join(tmpdir, 'emg3d_out.log')
        assert cfg['files']['save'] is False
        assert cfg['files']['load'] is False

        # Provide file names without save/load
        args_dict = self.args_dict.copy()
        args_dict['survey'] = 'test.h5'
        args_dict['model'] = 'unkno.wn'
        args_dict['output'] = 'out.npz'
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert cfg['files']['survey'] == join(tmpdir, 'test.h5')
        assert cfg['files']['model'] == join(tmpdir, 'unkno.h5')
        assert cfg['files']['output'] == join(tmpdir, 'out.npz')
        assert cfg['files']['save'] is False
        assert cfg['files']['load'] is False

        # Provide file names with save/load
        args_dict = self.args_dict.copy()
        args_dict['survey'] = 'test.h5'  # will be removed
        args_dict['model'] = 'unkno.wn'  # will be removed
        args_dict['output'] = 'out.npz'
        args_dict['save'] = 'test.npz'
        args_dict['load'] = 'test.npz'
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict.copy())
        assert cfg['files']['survey'] == join(tmpdir, 'test.h5')
        assert cfg['files']['model'] == join(tmpdir, 'unkno.h5')
        assert cfg['files']['output'] == join(tmpdir, 'out.npz')
        assert cfg['files']['save'] == join(tmpdir, 'test.npz')
        assert cfg['files']['load'] == join(tmpdir, 'test.npz')
        # Check save/load through cache with clean
        args_dict['model'] = 'model.h5'  # will NOT be removed
        args_dict['cache'] = 'new.npz'
        args_dict['clean'] = True
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert cfg['files']['model'] == join(tmpdir, 'model.h5')
        assert cfg['files']['save'] == join(tmpdir, 'new.npz')
        assert cfg['files']['load'] == join(tmpdir, 'new.npz')

        # .-trick.
        args_dict = self.args_dict.copy()
        args_dict['config'] = '.'
        _, term = cli.parser.parse_config_file(args_dict)
        assert term['config_file'] == '.'

        # Not existent.
        args_dict = self.args_dict.copy()
        args_dict['config'] = 'bla'
        _, term = cli.parser.parse_config_file(args_dict)
        assert sep + 'bla' in term['config_file']

    def test_term_various(self, tmpdir):

        args_dict = self.args_dict.copy()
        args_dict['nproc'] = -1
        args_dict['verbosity'] = 20
        args_dict['dry_run'] = True
        args_dict['layered'] = True
        args_dict['clean'] = True
        args_dict['gradient'] = True
        args_dict['path'] = tmpdir
        args_dict['survey'] = 'testit'
        args_dict['model'] = 'model.json'
        args_dict['output'] = 'output.npz'
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert term['verbosity'] == 2  # Maximum 2!
        assert term['dry_run'] is True
        assert term['clean'] is True
        assert term['function'] == 'gradient'
        assert cfg['simulation_options']['max_workers'] == 1
        assert cfg['simulation_options']['layered'] is True
        assert cfg['files']['survey'] == join(tmpdir, 'testit.h5')
        assert cfg['files']['model'] == join(tmpdir, 'model.json')
        assert cfg['files']['output'] == join(tmpdir, 'output.npz')
        assert cfg['files']['log'] == join(tmpdir, 'output.log')

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
            f.write("save=test")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert cfg['files']['survey'] == join(tmpdir, 'testit.json')
        assert cfg['files']['model'] == join(tmpdir, 'thismodel.h5')
        assert cfg['files']['output'] == join(tmpdir, 'results.npz')
        assert cfg['files']['log'] == join(tmpdir, 'results.log')
        assert cfg['files']['save'] == join(tmpdir, 'test.h5')

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

        # Write a config file with load.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[files]\n")
            f.write(f"path={tmpdir}\n")
            f.write("survey=testit.json\n")
            f.write("model=thismodel\n")
            f.write("output=results.npz\n")
            f.write("save=test\n")
            f.write("load=test")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        assert cfg['files']['survey'] == join(tmpdir, 'testit.json')
        assert cfg['files']['model'] == join(tmpdir, 'thismodel.h5')
        assert cfg['files']['output'] == join(tmpdir, 'results.npz')
        assert cfg['files']['log'] == join(tmpdir, 'results.log')
        assert cfg['files']['save'] == join(tmpdir, 'test.h5')
        assert cfg['files']['load'] == join(tmpdir, 'test.h5')

    def test_simulation(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[simulation]\n")
            f.write("max_workers=5\n")
            f.write("gridding=fancything\n")
            f.write("name=PyTest simulation\n")
            f.write("file_dir=here\n")
            f.write("min_offset=1320\n")
            f.write("max_offset=5320\n")
            f.write("mean_noise=1.0\n")
            f.write("ntype=gaussian_uncorrelated")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        with pytest.warns(FutureWarning, match='in their own section'):
            cfg, term = cli.parser.parse_config_file(args_dict)
        sim_opts = cfg['simulation_options']
        noise_kwargs = cfg['noise_kwargs']
        assert sim_opts['max_workers'] == 5
        assert sim_opts['gridding'] == 'fancything'
        assert sim_opts['name'] == "PyTest simulation"
        assert noise_kwargs['min_offset'] == 1320.0
        assert noise_kwargs['max_offset'] == 5320.0
        assert noise_kwargs['mean_noise'] == 1.0
        assert noise_kwargs['ntype'] == 'gaussian_uncorrelated'
        assert sim_opts['file_dir'] == 'here'
        with pytest.raises(KeyError, match="receiver_interpolation"):
            assert sim_opts['receiver_interpolation'] == 'linear'

        with pytest.raises(TypeError, match="Unexpected parameter in"):
            with open(config, 'a') as f:
                f.write("\nanother=True")
            args_dict = self.args_dict.copy()
            args_dict['config'] = config
            _ = cli.parser.parse_config_file(args_dict)

        # Ensure it sets interpolation to linear for gradient
        args_dict = self.args_dict.copy()
        args_dict['gradient'] = True
        cfg, term = cli.parser.parse_config_file(args_dict)
        sim_opts = cfg['simulation_options']
        assert sim_opts['receiver_interpolation'] == 'linear'

        # Ensure config file overrides that
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[simulation]\n")
            f.write("receiver_interpolation=cubic")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        args_dict['gradient'] = True
        cfg, term = cli.parser.parse_config_file(args_dict)
        sim_opts = cfg['simulation_options']
        assert sim_opts['receiver_interpolation'] == 'cubic'

    def test_layered(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[simulation]\n")
            f.write("layered=True\n")
            f.write("[layered]\n")
            f.write("method=prism\n")
            f.write("radius=3000\n")
            f.write("factor=1.2\n")
            f.write("minor=0.8\n")
            f.write("check_foci=True\n")
            f.write("merge=False")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        args_dict['layered'] = None
        cfg, term = cli.parser.parse_config_file(args_dict)
        sim_opts = cfg['simulation_options']
        lay_opts = sim_opts['layered_opts']
        assert sim_opts['layered'] is True
        assert lay_opts['ellipse']['radius'] == 3000.0
        assert lay_opts['ellipse']['factor'] == 1.2
        assert lay_opts['ellipse']['minor'] == 0.8
        assert lay_opts['ellipse']['check_foci'] is True
        assert lay_opts['merge'] is False

        with pytest.raises(TypeError, match="Unexpected parameter in"):
            with open(config, 'a') as f:
                f.write("\nanother=True")
            args_dict = self.args_dict.copy()
            args_dict['config'] = config
            _ = cli.parser.parse_config_file(args_dict)

    def test_noise(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[noise_opts]\n")
            f.write("add_noise=True\n")
            f.write("min_offset=1320\n")
            f.write("max_offset=5320\n")
            f.write("mean_noise=1.0\n")
            f.write("ntype=gaussian_uncorrelated")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        noise_kwargs = cfg['noise_kwargs']
        assert noise_kwargs['add_noise']
        assert noise_kwargs['min_offset'] == 1320.0
        assert noise_kwargs['max_offset'] == 5320.0
        assert noise_kwargs['mean_noise'] == 1.0
        assert noise_kwargs['ntype'] == 'gaussian_uncorrelated'

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
            f.write("frequencies=f0")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)
        test = cfg['data']
        assert test['sources'] == ['Tx11']
        assert test['receivers'] == ['Rx1', 'Rx2']
        assert test['frequencies'] == ['f0']

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
            f.write("lambda_from_center=True\n")
            f.write("mapping=LnResistivity\n")
            f.write("vector=yz\n")
            f.write("domain=-2000, 2000; None; -4000, 1.111\n")
            f.write("stretching=1.1, 2.0\n")
            f.write("min_width_limits=20, 40; 30, 60; None\n")
            f.write("properties=0.3, 1, 2, 3, 4, 5, 6\n")
            f.write("center=0, 0, -500\n")
            f.write("cell_number=20, 40, 80, 100\n")
            f.write("min_width_pps=2, 3, 4\n")
            f.write("expand=1, 2\n")
            f.write("center_on_edge=False; False; True")

        args_dict = self.args_dict.copy()
        args_dict['config'] = config
        cfg, term = cli.parser.parse_config_file(args_dict)

        check = cfg['simulation_options']['gridding_opts']

        assert check['verb'] == 3
        assert check['frequency'] == 7.77
        assert check['seasurface'] == -200
        assert check['max_buffer'] == 5000
        assert check['lambda_factor'] == 0.8
        assert check['lambda_from_center']
        assert check['mapping'] == 'LnResistivity'
        assert check['vector'] == 'yz'
        assert check['domain']['x'] == [-2000.0, 2000.0]
        assert check['domain']['y'] is None
        assert check['domain']['z'] == [-4000.0, 1.111]
        assert check['stretching'] == [1.1, 2.0]
        assert check['min_width_limits']['x'] == [20, 40]
        assert check['min_width_limits']['y'] == [30, 60]
        assert check['min_width_limits']['z'] is None
        assert check['properties'] == [0.3, 1, 2, 3, 4, 5, 6]
        assert check['center'] == [0, 0, -500]
        assert check['cell_number'] == [20, 40, 80, 100]
        assert check['min_width_pps'] == [2, 3, 4]
        assert check['expand'] == [1, 2]
        assert check['center_on_edge']['x'] is False
        assert check['center_on_edge']['y'] is False
        assert check['center_on_edge']['z'] is True

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
            'layered': False,
            'path': None,
            'survey': 'survey.npz',
            'model': 'model.npz',
            'output': 'output.npz',
            'save': None,
            'load': None,
            'cache': None,
            'clean': False,
            'verbosity': 0,
            'dry_run': True,
            }

    if xarray is not None:
        # Create a tiny dummy survey.
        data = np.ones((1, 17, 1))
        data[0, 8:11, 0] = np.nan
        sources = emg3d.TxElectricDipole((4125, 4000, 4000, 0, 0))
        receivers = emg3d.surveys.txrx_coordinates_to_dict(
                emg3d.RxElectricPoint,
                (np.arange(17)*250+2000, 4000, 3950, 0, 0))
        survey = emg3d.Survey(
            name='CLI Survey',
            sources=sources,
            receivers=receivers,
            frequencies=1,
            noise_floor=1e-15,
            relative_error=0.05,
            data=data,
        )

    # Create a dummy grid and model.
    xx = np.ones(16)*500
    grid = emg3d.TensorMesh([xx, xx, xx], origin=np.array([0, 0, 0]))
    model = emg3d.Model(grid, 1.)
    model_vti = emg3d.Model(grid, 1., property_z=2.0)
    model_tri = emg3d.Model(grid, 1., 1.5, 2.0)

    def test_basic(self, tmpdir):

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

        with pytest.raises(SystemExit) as e:
            cli.run.simulation(args_dict)
        assert e.type == SystemExit
        assert "* ERROR   :: Config file not found: " in e.value.code

        # Missing survey.
        args_dict = self.args_dict.copy()
        args_dict['path'] = tmpdir
        args_dict['survey'] = 'wrong'
        with pytest.raises(SystemExit) as e:
            cli.run.simulation(args_dict)
        assert e.type == SystemExit
        assert "* ERROR   :: Survey file not found: " in e.value.code
        assert "wrong" in e.value.code

        # Missing model.
        args_dict = self.args_dict.copy()
        args_dict['path'] = tmpdir
        args_dict['model'] = 'phantommodel'
        with pytest.raises(SystemExit) as e:
            cli.run.simulation(args_dict)
        assert e.type == SystemExit
        assert "* ERROR   :: Model file not found: " in e.value.code
        assert "phantommodel" in e.value.code

        # Missing output directory.
        args_dict = self.args_dict.copy()
        args_dict['path'] = tmpdir
        args_dict['output'] = join('phantom', 'output', 'dir.npz')
        args_dict['save'] = join('phantom', 'simulation', 'save.npz')
        with pytest.raises(SystemExit) as e:
            cli.run.simulation(args_dict)
        assert e.type == SystemExit
        assert "* ERROR   :: Output directory does not exist: " in e.value.code
        assert join("phantom", "output") in e.value.code
        assert join("phantom", "simulation") in e.value.code

    def test_gradient_shape_anisotropy(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[gridding_opts]\n")
            f.write("center_on_edge=True\n")

        self.survey.to_file(os.path.join(tmpdir, 'survey.npz'), verb=0)

        fname = os.path.join(tmpdir, 'model.npz')
        saveinp = {'fname': fname, 'mesh': self.grid, 'verb': 0}

        args_dict = self.args_dict.copy()
        args_dict['path'] = tmpdir
        args_dict['config'] = os.path.join(tmpdir, 'emg3d.cfg')
        args_dict['verbosity'] = -1

        # isotropic
        emg3d.save(model=self.model, **saveinp)
        cli.run.simulation(args_dict.copy())
        iso = emg3d.load(os.path.join(tmpdir, 'output.npz'))
        assert_allclose(iso['gradient'].shape, self.model.shape)

        # VTI
        emg3d.save(model=self.model_vti, **saveinp)
        cli.run.simulation(args_dict.copy())
        vti = emg3d.load(os.path.join(tmpdir, 'output.npz'))
        assert_allclose(vti['gradient'].shape, (2, *self.model.shape))

        # tri-axial
        emg3d.save(model=self.model_tri, **saveinp)
        cli.run.simulation(args_dict.copy())
        tri = emg3d.load(os.path.join(tmpdir, 'output.npz'))
        assert_allclose(tri['gradient'].shape, (3, *self.model.shape))

    def test_run(self, tmpdir):

        # Write a config file.
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[files]\n")
            f.write("save=mysim.npz\n")
            f.write("[solver_opts]\n")
            f.write("sslsolver=False\n")
            f.write("semicoarsening=False\n")
            f.write("linerelaxation=False\n")
            f.write("maxit=1\n")
            f.write("[gridding_opts]\n")
            f.write("center_on_edge=True\n")

        # Store survey and model.
        self.survey.to_file(os.path.join(tmpdir, 'survey.npz'), verb=1)
        emg3d.save(os.path.join(tmpdir, 'model.npz'), model=self.model_vti,
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
        # Double-check VTI gradient.
        assert_allclose(res2['gradient'].shape, (2, *self.model.shape))
        # Assert we can load the simulation
        emg3d.Simulation.from_file(os.path.join(tmpdir, 'mysim.npz'))
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

        # Redo for misfit, loading existing simulation, setting to layered.
        args_dict = self.args_dict.copy()
        args_dict['config'] = os.path.join(tmpdir, 'emg3d.cfg')
        args_dict['path'] = tmpdir
        args_dict['forward'] = False
        args_dict['misfit'] = True
        args_dict['gradient'] = False
        args_dict['dry_run'] = False
        args_dict['layered'] = True  # Change to layered!
        args_dict['load'] = 'mysim.npz'
        args_dict['output'] = 'output3.npz'
        cli.run.simulation(args_dict)
        with open(os.path.join(tmpdir, 'output3.log'), 'r') as f:
            log = f.read()
        assert "Change «layered» of simulation to True." in log
        assert "Gridding: layered computation using method 'cylinder'" in log
        res3 = emg3d.load(os.path.join(tmpdir, 'output3.npz'))
        assert 'misfit' in res3
        assert 'gradient' not in res3

    def test_data(self, tmpdir):

        # Write a config file; remove_empty=False (default)
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[gridding_opts]\n")
            f.write("center_on_edge=True\n")
            f.write("[data]\n")
            f.write("sources=TxED-1\n")
            f.write("receivers=RxEP-05, RxEP-10, RxEP-02, RxEP-12, RxEP-06\n")
            f.write("frequencies=f-1\n")

        # Store survey and model.
        self.survey.to_file(os.path.join(tmpdir, 'survey.npz'), verb=1)
        emg3d.save(os.path.join(tmpdir, 'model.npz'), model=self.model,
                   mesh=self.grid, verb=1)

        # Run a dry run (to output.npz).
        args_dict = self.args_dict.copy()
        args_dict['config'] = os.path.join(tmpdir, 'emg3d.cfg')
        args_dict['path'] = tmpdir
        cli.run.simulation(args_dict.copy())

        # Ensure dry_run returns same shaped data as the real thing.
        res = emg3d.load(os.path.join(tmpdir, 'output.npz'))
        assert_allclose(res['data'].shape, (1, 5, 1))
        assert res['n_observations'] == 4

        # Append config file with: remove_empty=True
        with open(config, 'a') as f:
            f.write("\nremove_empty=True")

        # Run a dry run (to output.npz).
        cli.run.simulation(args_dict)

        # Ensure dry_run returns same shaped data as the real thing.
        res = emg3d.load(os.path.join(tmpdir, 'output.npz'))
        assert_allclose(res['data'].shape, (1, 4, 1))
        assert res['n_observations'] == 4

    def test_expand(self, tmpdir):

        # Write a config file; remove_empty=False (default)
        config = os.path.join(tmpdir, 'emg3d.cfg')
        with open(config, 'w') as f:
            f.write("[gridding_opts]\n")
            f.write("center_on_edge=True\n")
            f.write("expand=2, 3\n")
            f.write("seasurface=8000.0\n")

        # Store survey and model.
        self.survey.to_file(os.path.join(tmpdir, 'survey.npz'), verb=1)
        emg3d.save(os.path.join(tmpdir, 'model.npz'), model=self.model,
                   mesh=self.grid, verb=1)

        # Run a dry run to store simulation.
        args_dict = self.args_dict.copy()
        args_dict['config'] = os.path.join(tmpdir, 'emg3d.cfg')
        args_dict['path'] = tmpdir
        args_dict['save'] = 'simulation1.h5'
        with pytest.warns(FutureWarning, match='A property-complete'):
            cli.run.simulation(args_dict.copy())

        # Replace / add dicts
        s = emg3d.Simulation.from_file(os.path.join(tmpdir, 'simulation1.h5'))
        s._dict_efield = {'bla': 1.0}
        s._dict_bfield = {'2': 3.4}
        s.to_file(os.path.join(tmpdir, 'simulation1.h5'))

        # Run a second dry run loading existing with clean; use other model.
        emg3d.save(os.path.join(tmpdir, 'model.npz'), model=self.model_vti,
                   mesh=self.grid, verb=1)
        args_dict['model'] = 'model.npz'
        args_dict['load'] = 'simulation1.h5'
        args_dict['save'] = 'simulation2.h5'
        args_dict['clean'] = True
        cli.run.simulation(args_dict.copy())

        s1 = emg3d.Simulation.from_file(os.path.join(tmpdir, 'simulation1.h5'))
        s2 = emg3d.Simulation.from_file(os.path.join(tmpdir, 'simulation2.h5'))

        # Ensure model has changed.
        assert s1.model.property_z is None
        assert s2.model.property_z is not None
        assert_allclose(s1.model.property_x[..., -1],
                        s2.model.property_z[..., -1])
        assert_allclose(s1.model.property_x[..., :-1]*2,
                        s2.model.property_z[..., :-1])
        assert s1._dict_efield == {'bla': 1.0}  # Made-up one
        assert s1._dict_bfield == {'2': 3.4}    # Made-up one
        assert s2._dict_efield == {'TxED-1': {'f-1': None}}  # Re-created
        assert not hasattr(s2, '_dict_bfield')  # Deleted


@disable_numba()
@pytest.mark.script_launch_mode('subprocess')
def test_import_time(script_runner):
    # Relevant for responsiveness of CLI: How long does it take to import?
    cmd = ["python", "-Ximporttime", "-c", "import emg3d"]
    out = script_runner.run(cmd, print_result=False)
    import_time_s = float(out.stderr.split('|')[-2])/1e6
    # Currently we check t < 1.0 s.
    assert import_time_s < 1.0
