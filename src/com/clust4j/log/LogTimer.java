package com.clust4j.log;

import java.text.SimpleDateFormat;
import java.util.Date;

public class LogTimer implements Timer {
	private static final ThreadLocal<SimpleDateFormat> dateFormat = new ThreadLocal<SimpleDateFormat>(){
		@Override protected SimpleDateFormat initialValue() {
			return new SimpleDateFormat("dd-MMM HH:mm:ss.SSS");
		}
	};
	
	private static final ThreadLocal<SimpleDateFormat> shortFormat = new ThreadLocal<SimpleDateFormat>(){
		@Override protected SimpleDateFormat initialValue() {
			return new SimpleDateFormat("HH:mm:ss.SSS");
		}
	};
	
	public final long _start = System.currentTimeMillis();
	public final long _nanos = System.nanoTime();
	
	// Empty constructor
	public LogTimer(){}
	
	
	/**Return the difference between when the timer was created and the current time. */
	@Override public long time() { return System.currentTimeMillis() - _start; }
	@Override public long nanos(){ return System.nanoTime() - _nanos; }
	
	@Override
	public String toString() {
		final long now = System.currentTimeMillis();
		return LogTimeFormatter.millis(now - _start, false) + 
				" (Wall: " + dateFormat.get().format(new Date(now)) + ") ";
	}
	
	/** return the start time of this timer.**/
	@Override public String startAsString() { return dateFormat.get().format(new Date(_start)); }
	/** return the start time of this timer.**/
	@Override public String startAsShortString() { return shortFormat.get().format(new Date(_start)); }
}
