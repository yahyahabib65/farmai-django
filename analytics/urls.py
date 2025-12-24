from django.urls import path
from .views import DashboardView, dashboard_view

urlpatterns = [
    # This creates the link: http://yoursite.com/analytics/dashboard/1/
    path('dashboard/<int:farm_id>/', DashboardView.as_view()),
    path('dashboard/', dashboard_view, name='dashboard'),
]