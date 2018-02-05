The key ideas summarizing a set of sentences can be succinctly represented with the help
of short key-phrases that provide information about the theme(s) present in the targeted
text.  These key-phrases can be used as a means to provide users with information re-
garding the primary themes present in a set of documents and grouping similar phrases
pertaining to the same theme together can allow the users to query the theme that they
are interested in, without acknowledging the other topics.  This is of particular relevance
in the field of hospitality, for both providers and consumers.  For example, a hotel owner
may be interested in knowing all the bad reviews that their hotel’s staff has received
to  improve  the  quality  of  housekeeping  and  all  the  good  reviews  that  their  food  has
received in order to improve their menu.  Similarly, a customer may only be interested
in the kind of facilities the hotel has to offer and not the location or food.  In such cases,
the hotel owner should have access to all reviews pertaining only to staff/food while the
customer  should  be  able  to  view  all  reviews  pertaining  to  facilities  and  filter  out  the
others that she is not interested in.  However, since key-phrase extraction techniques do
not preserve the semantics or context of the phrase, grouping similar phrases can be a
challenging task.  In this project,  we summarize the methods explored in
order to successfully group similar phrases pertaining to a single theme together.  For all
the methods explained, by similarity, we mean phrases that may be worded differently
but occur in similar contexts and share an overlaying theme.  For example,  given the
phrases ‘cheese omlette’,  ‘breakfast buffet’,  ‘short ribs’ and ‘evening party’,  we try to
cluster the first three together into a single group and the last one into a separate group
