library(magrittr)
library(tk_tbl)

dat = read.table("~/dev/idr/idr_peaks-overlapped-peaks.txt", stringsAsFactors = F, check.names = F) %>% tk_tbl()
dat
dat = dat %>% filter(start1 < stop1, start2 < stop2, chr1 == chr2 )
dat
hist(dat$stop1 - dat$start1)
hist(dat$stop2 - dat$start2)
tibble(peak_width1 = dat$stop1 - dat$start1, peak_width2 = dat$stop2 - dat$start2 ) %>% ggplot(aes(peak_width1, peak_width2)) + geom_point(alpha=0.1)
plot(ecdf(dat$IDR))
result = dat %>%
filter(IDR <= 0.05) %>%
mutate(start = as.integer(.5 * (start1 + start2)),
stop = as.integer(.5 * (stop1 + stop2)),
sig = .5 * (sig.value1 + sig.value2) ) %>%
mutate(name=".") %>%
select(chr = chr1, start, stop, name, sig) %>%
arrange(chr, start)
result
