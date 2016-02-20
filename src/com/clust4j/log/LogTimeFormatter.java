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
	    
	    if( !truncate ) return String.format("%02d:%02d:%02d.%03d", slots.hr, slots.min, slots.sec, slots.ms);
	    if( slots.hr != 0 ) return String.format("%2d:%02d:%02d.%03d", slots.hr, slots.min, slots.sec, slots.ms);
	    if( slots.min != 0 ) return String.format("%2d min %2d.%03d sec", slots.min, slots.sec, slots.ms);
	    return String.format("%2d.%03d sec", slots.sec, slots.ms);
	}
	
	/**
	 * Adapted from H2O "PrettyPrint"
	 * @param micros
	 * @return
	 */
	/*
	public static String micros(long amt) {
		final TimeSlots slots = fromTimeUnit(amt, TimeUnit.MICROSECONDS);
		
	    if( slots.hr != 0 ) return String.format("%2d:%02d:%02d.%03d", slots.hr, slots.min, slots.sec, slots.ms);
	    if( slots.min != 0 ) return String.format("%2d min %2d.%03d sec", slots.min, slots.sec, slots.ms);
	    if( slots.sec != 0 ) return String.format("%2d.%03d sec", slots.sec, slots.ms);
	    if( slots.ms != 0 ) return String.format("%3d.%03d msec", slots.ms, slots.us);
	    return String.format("%3d usec", slots.us);
	}
	*/
	
	private static TimeSlots fromTimeUnit(long amt, final TimeUnit unit) {
		final long hr = unit.toHours(amt); 
		amt -= subtractAmt(hr, unit, TimeUnit.HOURS);
		
		final long min = unit.toMinutes(amt); 
	    amt -= subtractAmt(min, unit, TimeUnit.MINUTES);
	    
	    final long sec = unit.toSeconds(amt); 
	    amt -= subtractAmt(sec, unit, TimeUnit.SECONDS);
	    
	    final long ms = unit.toMillis(amt);
	    
	    /*if(unit.equals(TimeUnit.MICROSECONDS))
	    	amt -= TimeUnit.MILLISECONDS.toMicros(ms);
	    else*/
	    	amt = 0;
	    
	    return new TimeSlots(hr, min, sec, ms, amt);
	}
	
	private static long subtractAmt(long amt, TimeUnit unit, TimeUnit trans) {
		switch(unit) {
			case MILLISECONDS:
				return trans.toMillis(amt);
			/*
			case MICROSECONDS:
				return trans.toMicros(amt);
			*/
			default:
				throw new IllegalArgumentException(unit.toString()+" not supported");
		}
	}
}
