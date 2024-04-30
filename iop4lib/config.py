# other imports
import os, yaml, logging
from pathlib import Path
from importlib.resources import files
import matplotlib, matplotlib.pyplot

# Disable matplotlib logging except for warnings and above

matplotlib.pyplot.set_loglevel('warning')

# logging
import logging
logger = logging.getLogger(__name__)

# Set journal_mode=WAL and synchronous=NORMAL for sqlite3 databases on 
# connection, to improve support for concurrent write access to the 
# database during parallel reduction

from django.db.backends.signals import connection_created
from django.dispatch.dispatcher import receiver

@receiver(connection_created)    
def enable_wal_sqlite(sender, connection, **kwargs) -> None:
    if connection.vendor == "sqlite":
        with connection.cursor() as cursor:
            cursor.execute('PRAGMA synchronous=NORMAL;')
            cursor.execute('PRAGMA journal_mode=WAL;')

class Config(dict):
    r""" Configuration class for IOP4.
    
    This class is a singleton, that is, it always returns the same instance of the class.
    """

    # PATH CONFIG

    basedir = Path(__file__).parents[1].absolute()

    # ALL OTHER CONFIG OPTIONS READ FROM SEPARATE CONFIG FILE (see config.example.yaml)

    _instance = None
    _configured = False

    def __new__(cls, *args, **kwargs):
        """ returns always the same instance of the class. """
        if Config._instance is None:
            Config._instance =  super().__new__(cls)
        return Config._instance
        
    def __init__(self, reconfigure=False, **kwargs):
        """
        Reads the specified config file, or the default one, and creates a configuration object.
        
        The configuration is performed only once, the first time it is called, unless you pass reconfigure=True.

        Parameters
        ----------
        config_path: str, default None
            Path to the configuration file. If None, it will try to use, in order: 
            1. The current config file
            2. The file specified by the environment variable IOP4_CONFIG_FILE 
            3. ~/.iop4.config.yaml, if it exists
            4. The example config file in the iop4lib package.
        config_db : bool, default False
            If True, configures django ORM to make the models and database accesible. It 
            should be used only once the top level of your script.
        gonogui : bool, default True
            If True, configures matplotlib to work without a GUI.
        jupytermode : bool, default False
            Code in jupyter notebooks is actually executed in an async context, which django ORM tries to
            protect against. This async context should in principle be disabled running '%autoawait off' in
            jupyter, but this might fail.

            If jupytermode is True, iop4lib will configure django ORM to disable this protection (setting
            the environment variable DJANGO_ALLOW_ASYNC_UNSAFE to true). You should try first to run '%autoawait off'.

            For IPython, you should try also '%autowait off' first too. Moreover, it seems that is not necessary to 
            use jupytermode=True in IPython Shell, even without running '%autowait off'.

        Returns
        -------
        self

        Attributes
        ----------
        Different configuration parameters such as database path, log format, credentials 
        for data download from remote sources, etc.
        """

        super(Config, self).__init__(**kwargs)
        self.__dict__ = self

        if reconfigure or not Config._configured:
            self.configure(**kwargs)

    def copy(self, iop4conf):
        for k, v in iop4conf.items():
            setattr(self, k, v)

    def configure(self, config_path=None, config_db=False, config_logging=True, gonogui=True, jupytermode=False, **kwargs):
    
        Config._configured = True

        # If config_path is None, either use the one already in use or the default one

        if config_path is None:
            if hasattr(self, 'config_path') and self.config_path is not None:
                config_path = self.config_path
            else:

                if os.getenv("IOP4_CONFIG_FILE") is not None:
                    config_path = Path(os.getenv("IOP4_CONFIG_FILE")).expanduser()
                elif Path("~/.iop4.config.yaml").expanduser().exists():
                    config_path = Path("~/.iop4.config.yaml").expanduser()
                else:
                    config_path = files("iop4lib") / "config.example.yaml"

                if not config_path.exists():
                    raise FileNotFoundError(f"Config file {config_path} not found.")
        
        self.config_path = Path(config_path).expanduser()

        # Load config file and set attributes

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        for k, v in config_dict.items():
            setattr(self, k, v)

        # Override with config options in the environment
        for k, v in os.environ.items():
            if k.startswith("IOP4_") and k != "IOP4_CONFIG_FILE":
                setattr(self, k[5:].lower(), v)

        # Override with config options passed as kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Allow paths relative to home directory

        for k, v in self.items():
            if (k in ['basedir', 'datadir', 'log_file'] or 'path' in k) and v is not None: # config variables should have ~ expanded
                setattr(self, k, str(Path(v).expanduser()))

        # If data dir does not exist, create it
        
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)

        # Check if the logs subdirectory exists, if not, create it
        
        if not os.path.exists(os.path.join(self.datadir, "logs")):
            os.makedirs(os.path.join(self.datadir, "logs"))

        # Load OSN names from external file if indicated, load them into patterns like name*.fit, name*.fits, or name*.fts.

        if self.osn_source_list_path is not None and os.path.exists(self.osn_source_list_path):
            with open(self.osn_source_list_path, 'r') as f:
                self.osn_fnames_patterns += [fr"(^{s[:-1]}.*\.fi?ts?$)" for s in f.readlines() if s[0] != '#']

        if config_logging:
            self.configure_logging()
            
        if gonogui:
            matplotlib.use("Agg")

        if jupytermode:
            os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

        if config_db:
            self.configure_db()

    def configure_db(self):

        import django
        from django.conf import settings
        from django.apps import apps
        
        settings.configure(
            INSTALLED_APPS=[
                'iop4api',
            ],
            DATABASES = {
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": self.db_path,
                }
            },
            DEBUG = False,
            LOGGING_CONFIG = None, # otherwise breaks logging colors inside docs notebooks
            TEMPLATES = [ # template engine needed for the automatic summary emails
                {
                    'BACKEND': 'django.template.backends.django.DjangoTemplates',
                    'DIRS': [],
                },
            ],
        )
        
        apps.populate(settings.INSTALLED_APPS)
    
        django.setup()


    def configure_logging(self):
        r""" Adds the '%(proc_memory)' field in the log record."""
        old_factory = logging.getLogRecordFactory()

        def record_factory_w_proc_memory(*args, **kwargs):
            from iop4lib.utils import get_mem_current
            record = old_factory(*args, **kwargs)
            record.proc_memory = f"{get_mem_current()/1024**2:.0f} MB"
            return record

        if old_factory.__name__ == record_factory_w_proc_memory.__name__:
            return
        
        logging.setLogRecordFactory(record_factory_w_proc_memory)


    def is_valid(self):
        r""" Checks that the configuration file is correct by comparing it with the default one.
        
        Returns
        -------
        bool
            True if the configuration file is correct, False otherwise.

        """

        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        with open(files("iop4lib") / "config.example.yaml", 'r') as f:
            config_dict_example = yaml.safe_load(f)

        wrong = False

        for k, v in config_dict_example.items():
            if k not in config_dict:
                logger.error(f"ERROR: {k} missing in config file.")
                wrong = True

        return not wrong