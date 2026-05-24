from django.urls import path

from .views import (
    home,
    forecast
)

urlpatterns = [

    # =========================
    # HOME PAGE
    # =========================
    path(
        "",
        home,
        name="home"
    ),

    # =========================
    # FORECAST API
    # =========================
    path(
        "forecast/",
        forecast,
        name="forecast"
    ),
]