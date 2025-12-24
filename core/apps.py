from django.apps import AppConfig


class CoreConfig(AppConfig):
    name = 'core'
    verbose_name = 'Farm Management'

    def ready(self):
        from django.contrib import admin
        admin.site.site_header = "FarmAI Settings"
        admin.site.site_title = "FarmAI"
        admin.site.index_title = "Manage Your Farm Data"