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
package com.clust4j.except;

public class IllegalClusterStateException extends IllegalStateException {

	private static final long serialVersionUID = -2379108879459786857L;

	public IllegalClusterStateException() {
		super();
	}
	
	public IllegalClusterStateException(final String msg) {
		super(msg);
	}
	
	public IllegalClusterStateException(final Throwable thrown) {
		super(thrown);
	}
	
	public IllegalClusterStateException(final String msg, final Throwable thrown) {
		super(msg, thrown);
	}
}
