# other imports
import os, pathlib, yaml, logging

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

    .. note::
        OSN_SourceList.txt contains the starting characters of the filenames of the sources in the OSN
        repo. For each entry, it is loaded in the osn_fnames_patterns list as a regular 
        expression (^<entry>.*\.fit(?:s+)$), where <entry> is the entry in the file. This will match any 
        filename starting as the file entry and ending in either .fit or .fits.
        
    """

    # PATH CONFIG

    basedir = pathlib.Path(__file__).parents[1].absolute()

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
        The configuration is performed only once, the first time it is called.

        Parameters
        ----------
        config_path: str, default None
            Path to the configuration file. If None, the default configuration file is used.
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

        # If config_path is None, either use already in use or the default one

        if config_path is None:
            if hasattr(self, 'config_path') and self.config_path is not None:
                config_path = self.config_path
            else:
                config_path = pathlib.Path(self.basedir) / "config" / "config.yaml"

                if not config_path.exists():
                    config_path = pathlib.Path(self.basedir) / "config" / "config.example.yaml"
        
        self.config_path = config_path

        # Load config file and set attributes

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        for k, v in config_dict.items():
            setattr(self, k, v)

        # Override with config options passed as kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Allow paths relative to home directory

        for k, v in self.items():
            if (k in ['basedir', 'datadir', 'log_file'] or 'path' in k) and v is not None: # config variables should have ~ expanded
                setattr(self, k, str(pathlib.Path(v).expanduser()))

        # If data dir does not exist, create it
        
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)

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
        
        #import sys
        #sys.path.append(r"/path/to/iop4/iop4site/")
        #not necessary anymore, now app is a module.
        settings.configure(
            INSTALLED_APPS=[
                'iop4api',
                # # apps below not necessary for iop4lib but allows using User model from the iop4.py -i shell 
                # # (python manage.py shell includes this by default, since it uses the settings in iop4site.settings)
                # "iop4site.iop4site.apps.IOP4AdminConfig", 
                # "django.contrib.auth", 
                # "django.contrib.contenttypes",
                # "django.contrib.sessions",
                # "django.contrib.messages",
                # "django.contrib.staticfiles",
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

        with open(pathlib.Path(self.basedir) / "config" / "config.example.yaml", 'r') as f:
            config_dict_example = yaml.safe_load(f)

        wrong = False

        for k, v in config_dict_example.items():
            if k not in config_dict:
                logger.error(f"ERROR: {k} missing in config file.")
                wrong = True

        return not wrong