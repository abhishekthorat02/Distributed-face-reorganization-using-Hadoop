import java.io.IOException;
import java.util.Iterator;
import java.io.*;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class VidReducer extends Reducer < Text, Text, Text, Text > {

	public void reduce(Text key, Iterable < Text > values, Context context)
	throws IOException, InterruptedException {
		Iterator < Text > it = values.iterator();
		while (it.hasNext()) {
			context.write(key, it.next());
		}
	}

}
