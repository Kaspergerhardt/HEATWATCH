from . import apple, coros, fitbit, garmin_connect, garmin_fit, polar, strava

# brand in the brand overview -> readers that read that participant's folder
READERS = {
    "GarminC": [garmin_connect.parse],
    "GarminI": [garmin_fit.parse],
    "Garmin (both)": [garmin_connect.parse, garmin_fit.parse],
    "Garmin": [garmin_connect.parse, garmin_fit.parse],        # 'Garmin' without 'yes': let the files decide
    "Coros": [garmin_fit.parse, coros.parse],                  # .fit files, or the .fit messages as .csv
    "Apple": [apple.parse],
    "Fitbit": [fitbit.parse],
    "Polar": [polar.parse],
    "Strava": [strava.parse],                                  # no HR used from Strava
}
