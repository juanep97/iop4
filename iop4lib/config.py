import os, pathlib, yaml, logging

import matplotlib, matplotlib.pyplot
matplotlib.pyplot.set_loglevel('warning')



class Config():
    """ Configuration class for IOP4.
    
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
        
    def __init__(self, config_path=None, config_db=False, gonogui=True, jupytermode=False, reconfigure=False):
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

        if reconfigure or not Config._configured:
            Config._configured = True
            
            if config_path is None:
                config_path = pathlib.Path(Config.basedir) / "config" / "config.yaml"

            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            for k, v in config_dict.items():
                setattr(Config, k, v)

            if Config.osn_source_list_path is not None and os.path.exists(Config.osn_source_list_path):
                with open(Config.osn_source_list_path, 'r') as f:
                    Config.osn_fnames_patterns += [f"(^{s[:-1]}.*\.fits?$)" for s in f.readlines() if s[0] != '#']

            if gonogui:
                matplotlib.use("Agg")

            if jupytermode:
                os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

            if config_db:
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
                            "NAME": Config.db_path,
                        }
                    },
                    DEBUG = False,
                )
                
                apps.populate(settings.INSTALLED_APPS)
            
                django.setup()