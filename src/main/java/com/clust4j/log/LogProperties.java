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

import java.net.URI;
import java.net.URISyntaxException;

public class LogProperties {
	/**
	 * If we can't find their user name, then it'll just
	 * end up as "user"...
	 * @return
	 */
	public static String DEFAULT_ROOT() {
		String usr = System.getProperty("user.name");
		if(null == usr)
			usr = "";
		
		String usr2 = usr.replaceAll(" ", "_");
		if(usr2.length() == 0)
			usr2 = "user";
		
		return "/tmp/clust4j-" + usr2;
	}
	
	
	static String root = DEFAULT_ROOT();
	private static URI ROOT;
	
	static {
		try {
			ROOT = new URI(root);
		} catch(URISyntaxException e) {
			throw new RuntimeException("Invalid root: " + root + ", " + e.getMessage());
		}
	}
	
	public static URI getRoot() {
		return ROOT;
	}
}
