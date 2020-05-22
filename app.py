import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.stattools import adfuller
import pydeck as pdk


st.header("Fire insurance in California")


st.markdown("""

> _Forecasts are normally generated from historical patterns.  What if the
patterns are non-stationary?_

""")

@st.cache(persist=True)
def load_data(plot=True):

	df = gpd.read_file('data/a000000af.gdbtable')
	df = df[(df.YEAR_.notna()) & (df.YEAR_ != '')]
	df['YEAR'] = gpd.pd.to_numeric(df.YEAR_)
	return df


df = load_data()

### Map data from 1920 - 2020.  
### Turn on to re-render

# f, ax = plt.subplots()
# subdata = df[df.YEAR >= 1920]
# ax = subdata.plot(
# 	column='YEAR',
# 	cmap='OrRd',
# 	figsize=(15,15), 
# 	legend=False,
# 	legend_kwds={'orientation': "horizontal", "drawedges": False}
# )
# ax.set_axis_off()
# plt.savefig(
# 	'ca.jpg',
# 	quality=90,
# 	transparent=True,
# 	bbox_inches='tight'
# )

###
###

st.markdown("""

There are a lot of reasons why wildfire or flood insurance is difficult, if
not impossible, to offer without public intervention.  Non-stationary patterns
are one reason.  What happens when a once-in-a-lifetime flood happens every
five years?  There are many other reasons.  This is a primer for wildfires in
California, with a focus on the data required for policymakers to address fire
insurance provision.

The following chart illustrates the spatial extent of fires over the past
century. It seems like all of California has been directly impacted by fire,
aside from the agricultural land in the middle of the state and the desert in
the southeast. 

The data is assembled from a multi-agency, statewide database of fire history.
For CAL FIRE, timber fires 10 acres or greater, brush fires 30 acres and
greater, and grass fires 300 acres or greater are included. For the USFS,
there is a 10 acre minimum for fires since 1950. The dataset contains wildfire
history, prescribed burns and other fuel modification projects &mdash and
updated May, 2020.


""")

st.image(
	"ca.jpg", 
	use_column_width=True, 
	caption="Fires in CA, 1920-2020, darker red indicates more recent."
)


st.markdown("""

Now consider the patterns over time, by cause.  It's pretty clear that the
average and variation of annual burned area has increased over time.  

Some of this may be a result of user-input or official categorization. 
Consider, for example, the decline in **Miscellaneous** fires over time
&mdash; probably because there was better record keeping.  Still, the total
burned area has increased.

""")

cause_dict = {
	"Lightning": 1,
	"Equipment Use": 2,
	"Smoking": 3,
	"Campfire": 4,
	"Debris": 5,
	"Railroad": 6,
	"Arson": 7,
	"Playing with fire": 8,
	"Miscellaneous": 9,
	"Vehicle": 10,
	"Powerline": 11,
	"Unknown / Unidentified": 14,
	"Escaped Prescribed Burn": 18
}


window = st.slider(
	'Moving average window (years)',
	3, 20, 15
)

cause_option = st.selectbox(
	'Cause',
	["All"] + list(cause_dict.keys())
)

show_option = st.selectbox(
	'Show raw numbers',
	['no', 'yes']
)


if cause_option != 'All':
	df = df[df.CAUSE == cause_dict[cause_option]]


tot = df.groupby('YEAR')['GIS_ACRES'].sum()
tot = gpd.pd.DataFrame(tot).reset_index()
tot.columns = ["year", "acres"]

tot = tot[tot.year > 1910]
tot.year = pd.to_datetime(tot.year, format='%Y')



line_raw = alt.Chart(tot).mark_line(color="#FAFAFA").encode(
	x=alt.X('year:T', axis=alt.Axis(title="")),
	y=alt.Y(
		'acres:Q', axis=alt.Axis(title="Burned area (acres)"))
)


line_smooth = alt.Chart(tot).mark_line(
	color='#e45756'
).transform_window(
	rolling_mean='mean(acres)',
	frame=[-window, 0]
).encode(
	x=alt.X('year:T', axis=alt.Axis(title="")),
	y=alt.Y('rolling_mean:Q', axis=alt.Axis(title="Burned area (acres)"))
)

if show_option == 'yes':
	st.altair_chart(line_raw + line_smooth, use_container_width=True)
else:
	st.altair_chart(line_smooth, use_container_width=True)



dftest = adfuller(tot.year, autolag='t-stat')
pvalue = dftest[1]

if (1 - pvalue < 0.05):
	stationarity = "non-stationary"
else:
	stationarity = "stationary"

st.markdown("""

The non-stationarity of the time-series is pretty clear from visual inspection
alone.  We can run some statistical tests anyway, just to be fancy.

The time-series for acres burned from **%s** is **%s** at the 5-percent
significance level (p-value = %s), according to the Augmented Dickey-Fuller
test.  The results depend on the data quality, not just the information
content.

""" % (
		cause_option,
		stationarity,
		np.round(1-pvalue, 2)
	)

)


st.markdown("""

I actually really like the synopsis about non-stationary time series for
prediction from Investopedia:

> Using non-stationary time series data in financial models produces
unreliable and spurious results and leads to poor understanding and
forecasting. The solution to the problem is to transform the time series data
so that it becomes stationary. If the non-stationary process is a random walk
with or without a drift, it is transformed to stationary process by
differencing. On the other hand, if the time series data analyzed exhibits a
deterministic trend, the spurious results can be avoided by detrending.
Sometimes the non-stationary series may combine a stochastic and deterministic
trend at the same time and to avoid obtaining misleading results both
differencing and detrending should be applied, as differencing will remove the
trend in the variance and detrending will remove the deterministic trend.

We can start to examine the underlying causes for the change in fire patterns
over time, which may include fuel level and composition; human activity; and
climate change.

""")

st.subheader("Factors")

st.markdown("""

The Gridded Surface Meteorological dataset provides high spatial resolution
(~4-km) daily surface fields of temperature, precipitation, winds, humidity
and radiation across the contiguous United States from 1979. The dataset
blends the high resolution spatial data from PRISM with the high temporal
resolution data from the National Land Data Assimilation System (NLDAS) to
produce spatially and temporally continuous fields that lend themselves to
additional land surface modeling.

The following chart illustrates two variables calculated by the GSM,
aggregated for the state of California (1980-2020):

1. **Fire danger index** (`bi`).  The monthly mean National Fire Danger Rating System
(NFDRS) fire danger index.  The measure represents an ensemble of many
environmental datasets.

2. **Dead fuel moisture** (`fm100`). Dead fuel moisture responds solely to ambient
environmental conditions and is critical in determining fire potential. Dead
fuel moistures are classed by timelag. A fuel's timelag is proportional to its
diameter and is loosely defined as the time it takes a fuel particle to reach
2/3's of its way to equilibrium with its local environment. There are
different classes of dead fuels. We use the 100-h dead fuel category, computed
from 24-hour average boundary condition composed of day length, hours of rain,
and daily temperature/humidity ranges.

""")


## Convert individual years into a single dataset
## Uncomment script to resave

# filelist = ["data/%s.csv" % x for x in range(1980, 2021)]

# res = []
# for fname in filelist:
# 	tempdf = pd.read_csv(fname, usecols=["date", "bi", "fm100"])
# 	res.append(tempdf)

# nfdrs = pd.concat(res, ignore_index=True)
# nfdrs.datetime = pd.to_datetime(nfdrs.date, format='%Y-%m-%d')
# nfdrs.to_pickle("data/nfdrs.pkl")

##
##

nfdrs = pd.read_pickle("data/nfdrs.pkl")

nfdrs_window = st.slider(
	'Symmetric moving average window (days on either side)',
	120, 600, 200
)

nfdrs_option = st.selectbox(
	'Show raw numbers',
	['True', 'False']
)

nfdrs_var = st.selectbox(
	'Variable',
	['fm100', 'bi']
)


# vis parameters
if nfdrs_option == 'False':
	vis = {'fm100': [8, 18], 'bi': [20, 50]}
else:
	vis = {'fm100': [0, 30], 'bi': [0, 80]}


nfdrs_data = alt.Chart(nfdrs).mark_circle(
	color="#A9BEBE", 
	size=1
).encode(
	x='date:T',
	y=nfdrs_var
)

nfdrs_smooth = alt.Chart(nfdrs).mark_line(
	color='#e45756'
).transform_window(
	rolling_mean='mean(%s)' % nfdrs_var,
	frame=[-nfdrs_window, nfdrs_window]
).encode(
	x=alt.X(
		'date:T',
		axis=alt.Axis(
			title=""
		),
	),
	y=alt.Y(
		'rolling_mean:Q', 
		scale=alt.Scale(domain=vis[nfdrs_var]),
		axis=alt.Axis(
			title="%s" % nfdrs_var
		)
	)
)


if nfdrs_option == 'True':
	st.altair_chart(nfdrs_data + nfdrs_smooth, use_container_width=True)
else:
	st.altair_chart(nfdrs_smooth, use_container_width=True)

st.markdown("""

**There is a statistically significant time trend for both variables.** 
Aggregate fuel moisture has decreased by roughly 15 percent since 1980 and the
burn index has increased by roughly 15 percent.

""")

st.subheader("Economic impact")

st.markdown("""

The pace of recovery has implications for insurance coverage and premiums. 
The following web map shows the location and value of residences that were
destroyed in the 2017 Santa Rosa wildfire.  The green locations are residences
that are in the process of being rebuilt &mdash; either completed, permitted,
or under construction.  The size of the dot corresponds to the pre-fire
valuation according to Zillow.  Hover over the dots to see the address and
value.

An animated version of the web map is
[here](https://s3-us-west-1.amazonaws.com/embeddable.earthrise.media/santa_rosa_rebuilding/index.html).


""")


burnt = pd.read_csv(
	"data/ALL_burnt_homes.csv", 
	usecols=["address", "lat", "lon", "zestimate"]
)

recovered = pd.read_csv(
	"data/ALL_recovered_homes.csv", 
	usecols=["address"]
)

recovered['status'] = "recovered"

homes = pd.merge(burnt, recovered, how='left', on=['address'])
homes.status[homes.status != 'recovered'] = 'destroyed'
homes.zestimate = np.round(homes.zestimate, 0)

layer = pdk.Layer(
	"ScatterplotLayer",
	homes,
	pickable=True,
	opacity=0.1,
	radius_scale=3,
	radius_min_pixels=1,
	radius_max_pixels=6,
	get_position=['lon', 'lat'],
	get_radius="zestimate/100000",
	get_fill_color="status == 'destroyed' ? [228, 87, 86] : [86, 228, 87]"
)


st.pydeck_chart(
	pdk.Deck(
		map_style = 'mapbox://styles/mapbox/dark-v9',
		initial_view_state = pdk.ViewState(
			latitude=38.4354,
			longitude=-122.65,
			zoom=10
		),
		layers=[layer],
		tooltip={"text": "{address}\n valuation: ${zestimate}"}
	)
)

destroyed_value = int(np.sum(homes.zestimate[homes.status == 'destroyed']))
recovered_value = int(np.sum(homes.zestimate[homes.status == 'recovered']))

nrow, ncol = homes.shape

rough_buildings = int(np.round(nrow / 10, 0) * 10)

st.markdown("""

There were roughly **%s buildings** destroyed, with a total value of **$%sB**.
The total value of buildings since rebuilt is **$%sB**, i.e., only **%s
percent** of the value has been recovered since November 2017.

""" % (
		"{:,}".format(rough_buildings),
		np.round((destroyed_value + recovered_value) / 1000000000, 2),
		np.round(recovered_value / 1000000000, 2),
		np.round(recovered_value / (recovered_value + destroyed_value), 3)*100
	)
)

