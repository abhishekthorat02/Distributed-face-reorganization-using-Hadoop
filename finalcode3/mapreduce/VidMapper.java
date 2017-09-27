import java.io.ByteArrayInputStream;
import java.io.*;
import java.io.IOException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;

public class VidMapper extends Mapper < Text, VideoObject, Text, Text > {

	private static final Log LOG = LogFactory.getLog(VidMapper.class);

	public void map(Text key, VideoObject value, Context context) throws IOException, InterruptedException {
		/*
		*/
		String folderref = key.toString().split("\\.")[0].replaceAll("[\\D]", "");			
		File dir = new File("/home/hduser/finalcode3/face_crop/testPics"+folderref);
        	dir.mkdir();
		File dir2 = new File("/home/hduser/finalcode3/face_crop/Cropped"+folderref);
		dir2.mkdir();
		String output = null;
		int totalsec = 0;
		//proc2.waitFor();
		int id = value.getId();
		ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(value.getVideoByteArray());
		LOG.info("Log__VideoConverter__byteArray: " + byteArrayInputStream.available());
		// wrote in local
		int flag = 0;
		int videonumber = 0;
		FileSystem hdfs = FileSystem.get(new Configuration());
		Path hdfsFilePath = new Path("/home/hduser/inputData/reference.jpg");
		Path localFilePath = new Path("/tmp/");
		hdfs.copyToLocalFile(hdfsFilePath, localFilePath);
		hdfsFilePath = new Path("/home/hduser/inputData/timer.txt");
		localFilePath = new Path("/tmp/");
		hdfs.copyToLocalFile(hdfsFilePath, localFilePath);
		//
		String fileName = key.toString();
		//Process inputData = Runtime.getRuntime().exec("hdfs dfs -get /home/hduser/inputData /tmp");
		//inputData.waitFor();
		LocalFileSystem fs = FileSystem.getLocal(context.getConfiguration());
		Path filePath = new Path("/tmp", fileName);
		Path resFile = new Path("/tmp", "res_" + fileName);
		System.out.println("File to Process :" + filePath.toString());
		FSDataOutputStream out = fs.create(filePath, true);
		output = filePath.toString().split("/")[2];
		out.write(value.getVideoByteArray());
		out.close();
		try {
			Process ffmpeg = Runtime.getRuntime().exec("avconv -i " + filePath.toString() + " -r 1 -f image2 /home/hduser/finalcode3/face_crop/testPics"+folderref+"/image-%3d.jpg");
			ffmpeg.waitFor();
			Process face_crop = Runtime.getRuntime().exec("python facecrop.py "+folderref+"", null, new File("/home/hduser/finalcode3/face_crop/"));
			face_crop.waitFor();
			ProcessBuilder builder = new ProcessBuilder("python", "create_csv.py",folderref);
			builder.directory(new File("/home/hduser/finalcode3/create_csv/"));
			builder.redirectOutput(new File("/home/hduser/finalcode3/create_csv/output/out"+folderref+".txt"));
			builder.redirectError(new File("/home/hduser/finalcode3/create_csv/output/outerror"+folderref+".txt"));
			Process p = builder.start(); // throws IOException
			p.waitFor();
			try (PrintWriter outline = new PrintWriter(new BufferedWriter(new FileWriter("/home/hduser/finalcode3/create_csv/output/out"+folderref+".txt", true)))) {
				outline.println("/tmp/reference.jpg;9999999");
			} catch (IOException e) {
				//exception handling left as an exercise for the reader
			}

			try (PrintWriter outpref = new PrintWriter(new BufferedWriter(new FileWriter("/home/hduser/finalcode3/create_csv/output/out"+folderref+".txt", true)))) {} catch (IOException e) {
				System.out.println("exception");
				e.printStackTrace();
				System.exit(-1);
			}
			System.setOut(new PrintStream(new FileOutputStream("/home/hduser/finalcode3/output/" + filePath.toString().split("/")[2] + ".txt")));
			try {
				String line;

				Process proc = Runtime.getRuntime().exec("./face /home/hduser/finalcode3/create_csv/output/out"+folderref+".txt 50.0", null, new File("/home/hduser/finalcode3/face_rec/"));
				BufferedReader in = new BufferedReader(
				new InputStreamReader(proc.getInputStream()));
				while ((line = in.readLine()) != null) {
					System.out.println(line);
				} in .close();
			} catch (Exception e) {
				// ...
			}


		} catch (IOException e) {
			System.out.println("exception");
			e.printStackTrace();
			System.exit(-1);
		}


		// mapper manupulation 
		try {
			Process duration = Runtime.getRuntime().exec(new String[] {
				"sh", "-c", "avconv -i " + filePath.toString() + " 2>&1 | grep 'Duration' | awk '{print $2}' | sed s/,//"
			});
			String durationline, finalstring;
			BufferedReader in = new BufferedReader(new InputStreamReader(duration.getInputStream()));
			durationline = in .readLine();
			int hour = Integer.parseInt(durationline.split(":")[0]);
			int min = Integer.parseInt(durationline.split(":")[1]);
			int sec = Integer.parseInt(durationline.split(":")[2].split("\\.")[0]);
			totalsec = hour * 3600 + min * 60 + sec; in .close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		try (BufferedReader brt = new BufferedReader(new FileReader("/tmp/timer.txt"))) {
			String timer;
			while ((timer = brt.readLine()) != null) {
				videonumber = Integer.parseInt(timer.toString());
				int data = Integer.parseInt(output.split("\\.")[0].replaceAll("[\\D]", ""));
				if (videonumber == data) {
					flag = 1;
				}
			}
		} catch (Exception e) {
			// code for execption
		}



		StringBuffer sb = new StringBuffer("");

		try (BufferedReader br = new BufferedReader(new FileReader("/home/hduser/finalcode3/output/" + output + ".txt"))) {
			String line, finaloutput;
			int seconds, finalseconds, hours, remainder, mins, secs;
			int multiply = Integer.parseInt(output.split("\\.")[0].replaceAll("[\\D]", ""));
			while ((line = br.readLine()) != null) {
				seconds = Integer.parseInt(line.toString().split("=")[1].replaceAll("[\\D]", ""));
				if (flag == 1) {
					finalseconds = seconds + videonumber;
				} else {
					finalseconds = seconds + (multiply * totalsec);
				}
				hours = (int) finalseconds / 3600;
				remainder = (int) finalseconds - hours * 3600;
				mins = remainder / 60;
				remainder = remainder - mins * 60;
				secs = remainder;

				finaloutput = hours + ":" + mins + ":" + secs + "\r\n ";
				if (!sb.toString().toLowerCase().contains(finaloutput.toLowerCase())) {
					sb.append(finaloutput);
				}
			}
		} catch (Exception e) {
			// code for execption
		}

		String outreducer = sb.toString();
		context.write(key, new Text(outreducer));
		sb.setLength(0);
		Process proc1 = Runtime.getRuntime().exec(new String[] {
			"sh", "-c", " rm -rf /home/hduser/finalcode3/face_crop/testPics"+folderref
		});
		proc1.waitFor();
		Process proc2 = Runtime.getRuntime().exec(new String[] {
			"sh", "-c", " rm -rf /home/hduser/finalcode3/face_crop/Cropped"+folderref
		});
		proc2.waitFor();
		Process proc3 = Runtime.getRuntime().exec(new String[] {
			"sh", "-c", " rm -rf "+filePath.toString()
		});
		proc3.waitFor();
	}

}
