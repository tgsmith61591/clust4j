/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.log;

import java.util.concurrent.TimeUnit;

public class LogTimeFormatter {
	final static class TimeSlots {
		final long hr;
		final long min;
		final long sec;
		final long ms;
		final long us;
		
		public TimeSlots(long hr, long min, long sec, long ms, long us) {
			this.hr = hr;
			this.min = min;
			this.sec = sec;
			this.ms = ms;
			this.us = us;
		}
	}
	
	/**
	 * Adapted from H2O "PrettyPrint"
	 * @param millis
	 * @param truncate
	 * @return
	 */
	public static String millis(long millis, boolean truncate) {
		final TimeSlots slots = fromTimeUnit(millis, TimeUnit.MILLISECONDS);
	    return millis(slots, truncate);
	}
	
	public static String millis(TimeSlots slots, boolean truncate) {
		if( !truncate ) return String.format("%02d:%02d:%02d.%03d", slots.hr, slots.min, slots.sec, slots.ms);
	    if( slots.hr != 0 ) return String.format("%2d:%02d:%02d.%03d", slots.hr, slots.min, slots.sec, slots.ms);
	    if( slots.min != 0 ) return String.format("%2d min %2d.%03d sec", slots.min, slots.sec, slots.ms);
	    return String.format("%2d.%03d sec", slots.sec, slots.ms);
	}
	
	private static TimeSlots fromTimeUnit(long amt, final TimeUnit unit) {
		final long hr = unit.toHours(amt); 
		amt -= subtractAmt(hr, unit, TimeUnit.HOURS);
		
		final long min = unit.toMinutes(amt); 
	    amt -= subtractAmt(min, unit, TimeUnit.MINUTES);
	    
	    final long sec = unit.toSeconds(amt); 
	    amt -= subtractAmt(sec, unit, TimeUnit.SECONDS);
	    
	    final long ms = unit.toMillis(amt);
	    amt = 0;
	    
	    return new TimeSlots(hr, min, sec, ms, amt);
	}
	
	static long subtractAmt(long amt, TimeUnit unit, TimeUnit trans) {
		switch(unit) {
			case MILLISECONDS:
				return trans.toMillis(amt);
			default:
				throw new UnsupportedOperationException(unit.toString()+" not supported");
		}
	}
}
