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

import java.text.SimpleDateFormat;
import java.util.Date;

import org.apache.commons.math3.util.FastMath;

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
	
	/**
	 * Formats the time differential between {@link #now()} and {@link #_start}
	 * @return the formatted time as a String
	 */
	public String formatTime() {
		return formatTime(now() - _start);
	}
	
	/**
	 * Formats the time differential between timeEnd and timeStart
	 * @param timeStart
	 * @param timeEnd
	 * @return the formatted time as a String
	 */
	public String formatTime(long timeStart, long timeEnd) {
		long te = FastMath.max(timeStart, timeEnd);
		long ts = FastMath.min(timeStart, timeEnd);
		return formatTime(te - ts);
	}
	
	public String formatTime(long millis) {
		return LogTimeFormatter.millis(millis, false);
	}
	
	@Override
	public String toString() {
		final long now = now();
		return LogTimeFormatter.millis(now - _start, false) + " " + wallMsg(now);
	}
	
	public String wallMsg() {
		return wallMsg(now());
	}
	
	private String wallMsg(final long now) {
		return "(Wall: " + wallTime(now) + ") ";
	}
	
	public String wallTime() {
		return wallTime(now());
	}
	
	public String wallTime(long now) {
		return dateFormat.get().format(new Date(now));
	}
	
	/** return the start time of this timer.**/
	@Override public String startAsString() { return dateFormat.get().format(new Date(_start)); }
	/** return the start time of this timer.**/
	@Override public String startAsShortString() { return shortFormat.get().format(new Date(_start)); }
	/** return the current time of this timer.**/
	@Override public String nowAsString() { return dateFormat.get().format(new Date(now())); }
	/** return the current time of this timer.**/
	@Override public String nowAsShortString() { return shortFormat.get().format(new Date(now())); }

	public long now() {
		return System.currentTimeMillis();
	}
}
